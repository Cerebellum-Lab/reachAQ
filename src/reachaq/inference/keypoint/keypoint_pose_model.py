import logging
import os
import typing

import numpy
import yaml

from ..pose_model import PoseModel

logger = logging.getLogger(__name__)


class KeypointPoseModel(PoseModel):
    """
    The live pose backend: a KeypointNet run in plain torch behind PoseModel.

    A model location is a directory written by keypoint.train:

        <location>/
            weights.pt              a state_dict, nothing pickled but tensors
            keypoint_pose.yaml

        weights: weights.pt
        imgsz: 256                  the frame size it was trained at
        network: {keypoints: 14, widths: [...], blocks: [...], neck: 64}
        bodyparts: [RH_flat, ...]   list, or dict of category -> [parts]

    The weights are loaded with weights_only, so a model directory cannot run
    code on load, and the network is rebuilt from the sidecar rather than
    unpickled. The names live in the sidecar for the same reason they did for
    YOLO: everything downstream is addressed by name, and the checkpoint's
    keypoint count is checked against them on load.

    Live frames go through a CUDA graph captured for the live batch: on the
    T1000 the forward of a network this size is dominated by per-layer launch
    overhead, which one graph replay removes. Any other batch runs eagerly.
    """

    DEFAULT_BODY_PART_CATEGORY = "single"
    SIDECAR = "keypoint_pose.yaml"
    DEFAULT_WEIGHTS = "weights.pt"

    def __init__(self, source: str, batch_size: int = 1, device: str = "cuda",
                 cuda_graph: bool = True):
        super().__init__()
        self._source = source
        self._batch_size = batch_size
        self._device = device
        self._want_graph = cuda_graph
        self._net = None
        self._graph = None
        self._graph_in = None
        self._graph_out = None
        self._imgsz = 256
        self._weights_path = None
        self._body_parts_count = 0

    # -- validation -------------------------------------------------------

    @classmethod
    def pre_validate(cls, location: str):
        """Cheap filesystem check, run in the parent before any import."""
        sidecar = os.path.join(location, cls.SIDECAR)
        if not os.path.isfile(sidecar):
            raise FileNotFoundError(f"{sidecar!r} does not exist")
        with open(sidecar, encoding="utf-8") as handle:
            spec = yaml.safe_load(handle) or {}
        weights = os.path.join(location, spec.get("weights", cls.DEFAULT_WEIGHTS))
        if not os.path.isfile(weights):
            raise FileNotFoundError(f"{weights!r} does not exist")
        if not spec.get("bodyparts"):
            raise ValueError(f"{sidecar!r} declares no bodyparts")
        if not isinstance(spec.get("network"), dict):
            raise ValueError(f"{sidecar!r} does not describe the network")

    @classmethod
    def has_trained_snapshot(cls, location: str) -> bool:
        try:
            cls.pre_validate(location)
        except (OSError, ValueError):
            return False
        return True

    # -- interface --------------------------------------------------------

    @property
    def supports_partial_batch(self) -> bool:
        """True: the network takes any batch, so the caller need not pad."""
        return True

    @property
    def crop_preserves_scale(self) -> bool:
        """False: every frame is resized to imgsz, as it was for training."""
        return False

    def is_valid(self) -> bool:
        try:
            self.pre_validate(self._source)
        except (OSError, ValueError):
            return False
        return True

    def load(self) -> None:
        import torch

        from .network import KeypointNet

        sidecar = os.path.join(self._source, self.SIDECAR)
        with open(sidecar, encoding="utf-8") as handle:
            spec = yaml.safe_load(handle) or {}
        self._imgsz = int(spec.get("imgsz", 256))
        self._weights_path = os.path.join(
            self._source, spec.get("weights", self.DEFAULT_WEIGHTS))
        self._load_body_parts(spec["bodyparts"])

        network = dict(spec["network"])
        if int(network.get("keypoints", -1)) != self._body_parts_count:
            raise ValueError(
                f"{sidecar!r} builds a network for {network.get('keypoints')} "
                f"keypoints but names {self._body_parts_count}")
        net = KeypointNet(**network)
        state = torch.load(self._weights_path, map_location="cpu", weights_only=True)
        # strict: a state_dict that does not fit the described network is a
        # mismatched sidecar, and loading it partially would run random weights.
        net.load_state_dict(state, strict=True)
        self._net = net.to(self._device).eval()

        self._warm_up()
        if self._want_graph:
            reason = self._graph_reason()
            if reason is None:
                self._capture_graph()
            else:
                logger.info("%s: graphed path unavailable: %s", self, reason)
        logger.info("%s: loaded %s, %d parts, imgsz %d", self,
                    os.path.basename(self._weights_path),
                    self._body_parts_count, self._imgsz)

    def prepare_live_batch(self, batch_size: int) -> None:
        """Capture the graph for the batch live inference sends.

        A CUDA graph only replays for the shape it was captured at, and the
        model is constructed for the padded offline batch, so without this the
        live batch would never match it and every live call would run eagerly.
        """
        if batch_size > 0:
            self._batch_size = int(batch_size)

    def runtime_detail(self) -> str:
        if self._graph is None:
            return (f"eager torch, imgsz {self._imgsz} "
                    f"(no CUDA graph: {self._graph_reason() or 'capture failed'})")
        return (f"CUDA graph, batch {self._batch_size}, imgsz {self._imgsz} "
                f"(eager for any other batch)")

    def predict(self, frames: numpy.ndarray) -> typing.List[numpy.ndarray]:
        """
        Pose for each frame, in input order.

        Returns (num_body_parts, 3) per frame - x, y, confidence - in
        full-frame pixel coordinates. Every part is always reported; an absent
        one comes back at low confidence, for the downstream gate to drop.
        """
        import torch

        tensor, rows = self._to_input(frames)
        if self._graph is not None and len(frames) == self._batch_size:
            self._graph_in.copy_(tensor)
            self._graph.replay()
            out = self._graph_out
        else:
            with torch.no_grad():
                out = self._decode(self._net(tensor))
        keypoints = out.clone() if out is self._graph_out else out
        if rows != self._imgsz:
            # x and y only; confidence has no unit. Pixel centres at integers,
            # so the scale is about the centre of the first pixel.
            gain = rows / self._imgsz
            keypoints[..., :2] = (keypoints[..., :2] + 0.5) * gain - 0.5
        return list(keypoints.cpu().numpy().astype("float32"))

    # -- internals --------------------------------------------------------

    def _to_input(self, frames):
        """(N, S, S, 3) uint8 -> (N, 1, imgsz, imgsz) float on the device.

        The cameras are monochrome; averaging the channels is exact for their
        frames and harmless for any other. A non-square frame is refused rather
        than stretched, because every model was trained on square ones.
        """
        import torch

        batch = numpy.asarray(frames)
        rows, cols = batch.shape[1:3]
        if rows != cols:
            raise ValueError(
                f"{rows}x{cols} frames cannot be scaled to the square imgsz "
                f"{self._imgsz} input without stretching them")
        tensor = torch.from_numpy(numpy.ascontiguousarray(batch)).to(
            self._device, non_blocking=True)
        if tensor.ndim == 4:
            tensor = tensor.float().mean(dim=3)
        tensor = tensor.float().div_(255.0).unsqueeze(1)
        if rows != self._imgsz:
            tensor = torch.nn.functional.interpolate(
                tensor, size=(self._imgsz, self._imgsz), mode="bilinear",
                align_corners=False)
        return tensor, rows

    @staticmethod
    def _decode(output):
        from .network import decode
        return decode(output)

    def _graph_reason(self) -> typing.Optional[str]:
        """Why the graphed path cannot be used, or None if it can."""
        if not str(self._device).startswith("cuda"):
            return "not on cuda"
        import torch
        if not torch.cuda.is_available():
            return "cuda unavailable"
        return None

    def _capture_graph(self) -> None:
        """Capture forward and decode once, so a live call is one replay."""
        import torch

        shape = (self._batch_size, 1, self._imgsz, self._imgsz)
        static = torch.zeros(shape, device=self._device, dtype=torch.float32)
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.no_grad():
                for _ in range(5):
                    self._decode(self._net(static))
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(graph):
                out = self._decode(self._net(static))
        except Exception as error:
            logger.warning("%s: graph capture failed (%s), running eagerly",
                           self, type(error).__name__)
            return
        self._graph = graph
        self._graph_in = static
        self._graph_out = out
        logger.info("%s: graphed path active, batch %d", self, self._batch_size)

    def _warm_up(self) -> None:
        """
        Run the model before anything is timed or waited on.

        The first calls allocate workspace and pick kernels - tens of
        milliseconds against a 6.67 ms frame period. Paying it in load() keeps
        it out of the live loop, where it would look like a dropped frame.
        """
        import torch

        blank = torch.zeros((1, 1, self._imgsz, self._imgsz), device=self._device)
        with torch.no_grad():
            for _ in range(3):
                self._net(blank)

    def _load_body_parts(self, declared) -> None:
        """Same ordering and grouping contract the other backends honour."""
        if isinstance(declared, dict):
            for category in declared.keys():
                self._body_part_categories.append(category)
                self._body_parts_by_category[category] = list()
            for category in self._body_part_categories:
                for part in declared[category]:
                    self._body_parts.append(part)
                    self._body_parts_by_category[category].append(part)
        elif isinstance(declared, list):
            self._body_part_categories.append(self.DEFAULT_BODY_PART_CATEGORY)
            self._body_parts_by_category[self.DEFAULT_BODY_PART_CATEGORY] = list()
            for part in declared:
                self._body_parts.append(part)
                self._body_parts_by_category[
                    self.DEFAULT_BODY_PART_CATEGORY].append(part)
        else:
            raise TypeError(
                f"Unhandled bodyparts type: {type(declared)}. value={declared}")
        self._body_parts_count = len(self._body_parts)

    def __str__(self):
        return f"KeypointPoseModel({os.path.basename(self._source)})"
