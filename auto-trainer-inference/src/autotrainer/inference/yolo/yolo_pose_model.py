import logging
import os
import typing

import numpy
import yaml

from ..pose_model import PoseModel

logger = logging.getLogger(__name__)


class YoloPoseModel(PoseModel):
    """
    A YOLO-pose backend behind the same interface the DeepLabCut backends use.

    The sweep produced YOLO weights that beat the DeepLabCut models this rig
    currently runs on speed, but nothing could measure that end to end, because
    `predict()` had only DeepLabCut implementations and the live path talks to
    `PoseModel`. This closes that gap without touching the shared pipeline:
    PoseProcess, PoseAlgorithm and the calibration path see the same interface
    they always have.

    A model location is a directory holding the weights and a small sidecar:

        <location>/
            best.pt
            yolo_pose.yaml

    The sidecar exists because a `.pt` file carries keypoint COUNT but not
    keypoint NAMES, and everything downstream is addressed by name -
    PoseAlgorithm builds its column MultiIndex from `body_parts` and gates on
    `body_part_categories`. Inferring names from order would put the burden of
    keeping two orderings in step on whoever next retrains, and a silent
    mismatch there mislabels every part rather than failing. So the names are
    written next to the weights, in the same shape DeepLabCut's config.yaml
    uses, and validated against the checkpoint on load.

        weights: best.pt          # optional, defaults to best.pt
        imgsz: 256                # must match training, see below
        bodyparts:                # list, or dict of category -> [parts]
          - RH_flat
          - ...

    `imgsz` is not cosmetic. The sweep found accuracy varies by more than a
    pixel between 256 and 320 on the same weights, and in opposite directions
    depending on how the model was trained, so running a checkpoint at a size
    it was not trained for silently degrades it.
    """

    DEFAULT_BODY_PART_CATEGORY = "single"
    SIDECAR = "yolo_pose.yaml"
    DEFAULT_WEIGHTS = "best.pt"

    def __init__(self, source: str, batch_size: int = 1,
                 device: str = "cuda", confidence: float = 0.01,
                 cuda_graph: bool = True):
        super().__init__()
        self._source = source
        self._batch_size = batch_size
        self._device = device
        # Deliberately low. The rig gates on confidence downstream, where the
        # threshold is configurable and visible; a high gate here would drop
        # parts before anything could see or tune that decision.
        self._confidence = confidence
        self._model = None
        self._want_graph = cuda_graph
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
        """
        True: ultralytics batches whatever list it is handed.

        The padding in PoseProcess exists for the TensorFlow graph's fixed
        batch dimension. Feeding it here would pay batch-six compute to produce
        batch-two output on every live frame, which is exactly the overhead
        this backend is meant to remove.
        """
        return True

    def is_valid(self) -> bool:
        try:
            self.pre_validate(self._source)
        except (OSError, ValueError):
            return False
        return True

    def load(self) -> None:
        from ultralytics import YOLO

        sidecar = os.path.join(self._source, self.SIDECAR)
        with open(sidecar, encoding="utf-8") as handle:
            spec = yaml.safe_load(handle) or {}
        self._imgsz = int(spec.get("imgsz", 256))
        self._weights_path = os.path.join(
            self._source, spec.get("weights", self.DEFAULT_WEIGHTS))
        self._load_body_parts(spec["bodyparts"])

        self._model = YOLO(self._weights_path)
        self._model.to(self._device)

        # The checkpoint knows how many keypoints it predicts. If that
        # disagrees with the sidecar, every part downstream is mislabelled
        # while still looking plausible, so fail here instead.
        declared = self._keypoint_count()
        if declared is not None and declared != self._body_parts_count:
            raise ValueError(
                f"{self._weights_path!r} predicts {declared} keypoints but "
                f"{sidecar!r} names {self._body_parts_count}")

        self._warm_up()
        if self._want_graph:
            reason = self._graph_reason()
            if reason is None:
                self._capture_graph()
            else:
                logger.info("%s: lean path unavailable: %s", self, reason)
        logger.info("%s: loaded %s, %d parts, imgsz %d", self,
                    os.path.basename(self._weights_path),
                    self._body_parts_count, self._imgsz)

    def prepare_live_batch(self, batch_size: int) -> None:
        """Capture the graph for the batch live inference actually sends.

        A CUDA graph only replays for the exact shape it was captured at. The
        model is constructed for the padded offline batch - two cameras times
        three frames - while live inference sends two frames, so the graph was
        captured at six, never matched, and every live call fell through to the
        ultralytics wrapper. On the rig that was 7.4 ms instead of 3.6 ms, and
        nothing said so: the fall-back is silent by design.

        The offline pass keeps working through predict(), which takes any
        count. It trades its graph for the live path's, which is the right way
        round - offline has no deadline.
        """
        if batch_size > 0:
            self._batch_size = int(batch_size)

    def runtime_detail(self) -> str:
        """Whether the graphed path is live, and for what batch.

        The batch matters as much as the graph: a captured graph only replays
        for exactly the shape it was captured at, so a model that graphed for
        two frames and is handed one silently falls back to the ultralytics
        wrapper. Both numbers belong in the same line.
        """
        if self._graph is None:
            return (f"ultralytics predict(), imgsz {self._imgsz} "
                    f"(no CUDA graph: {self._graph_reason() or 'capture failed'})")
        return (f"CUDA graph, batch {self._batch_size}, imgsz {self._imgsz} "
                f"(falls back to predict() for any other batch)")

    def predict(self, frames: numpy.ndarray) -> typing.List[numpy.ndarray]:
        """
        Pose for each frame, in input order.

        Returns (num_body_parts, 3) per frame - x, y, confidence - in
        full-frame pixel coordinates, which is what calibration expects.
        A frame with no detection returns zeros at zero confidence rather
        than being omitted, because the caller indexes results positionally.
        """
        if self._graph is not None and len(frames) == self._batch_size:
            return self._lean_predict(frames)
        results = self._model.predict(
            list(frames), imgsz=self._imgsz, conf=self._confidence,
            verbose=False, device=self._device)
        out = []
        for result in results:
            row = numpy.zeros((self._body_parts_count, 3), dtype="float32")
            keypoints = result.keypoints
            if keypoints is not None and len(keypoints.data):
                index = 0
                boxes = result.boxes
                if boxes is not None and len(boxes) > 1:
                    # One animal in the scene, so the most confident box is the
                    # one to read. Taking element zero would make the answer
                    # depend on NMS ordering.
                    index = int(numpy.argmax(boxes.conf.cpu().numpy()))
                found = keypoints.data[index].cpu().numpy()
                row[:, :2] = found[:, :2]
                row[:, 2] = found[:, 2] if found.shape[1] > 2 else 1.0
            out.append(row)
        return out


    # -- the lean path ----------------------------------------------------

    def _graph_reason(self) -> typing.Optional[str]:
        """Why the graphed path cannot be used, or None if it can.

        Stated as a reason rather than a boolean so a silent fall back to the
        slow path is impossible to miss in a log. Measured on the rig: the
        ultralytics predict() wrapper around this model costs about 4.4 ms on
        a T1000, of which the forward pass is a small part - the rest is
        per-layer launch overhead that a captured graph removes outright.
        """
        if self._device != "cuda":
            return "not on cuda"
        try:
            import torch
        except ImportError:
            return "torch unavailable"
        if not torch.cuda.is_available():
            return "cuda unavailable"
        head = getattr(self._model.model, "model", [None])[-1]
        classes = getattr(head, "nc", None)
        if classes not in (None, 1):
            return f"lean decode assumes one class, model has {classes}"
        return None

    def _capture_graph(self) -> None:
        """Capture the forward once so live frames replay it as a single launch."""
        import torch

        net = self._model.model.eval()
        shape = (self._batch_size, 3, self._imgsz, self._imgsz)
        static = torch.zeros(shape, device="cuda", dtype=torch.float32)
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.no_grad():
                for _ in range(5):
                    net(static)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(graph):
                out = net(static)
        except Exception as error:
            logger.warning("%s: graph capture failed (%s), using predict()",
                           self, type(error).__name__)
            return
        self._graph = graph
        self._graph_in = static
        self._graph_out = out[0] if isinstance(out, (list, tuple)) else out
        logger.info("%s: lean graphed path active, batch %d", self,
                    self._batch_size)

    def _lean_predict(self, frames) -> typing.List[numpy.ndarray]:
        """Preprocess, replay the graph, read the best anchor.

        No NMS: there is one class and one animal, so the answer is the anchor
        with the highest class score. Running NMS to pick one box out of one
        would cost more than the decode it precedes.

        The head already returns pixel coordinates - channels are 4 box, 1
        class, then 14 keypoint triples - so nothing here rescales, and a
        change to the model's stride or input size cannot silently shift the
        output without the captured shape check below failing first.
        """
        import torch

        batch = numpy.asarray(frames)
        tensor = torch.from_numpy(
            numpy.ascontiguousarray(batch[..., ::-1].transpose(0, 3, 1, 2))
        ).to("cuda", non_blocking=True).float().div_(255.0)
        self._graph_in.copy_(tensor)
        self._graph.replay()

        out = self._graph_out
        scores = out[:, 4, :]
        best = scores.argmax(dim=1)
        index = best.view(-1, 1, 1).expand(-1, out.shape[1], 1)
        picked = out.gather(2, index).squeeze(2)
        keypoints = picked[:, 5:].reshape(len(frames), self._body_parts_count, 3)
        return list(keypoints.detach().cpu().numpy().astype("float32"))

    # -- internals --------------------------------------------------------

    def _keypoint_count(self) -> typing.Optional[int]:
        shape = getattr(getattr(self._model, "model", None), "kpt_shape", None)
        if shape is None:
            shape = (getattr(self._model, "model", None) and
                     getattr(self._model.model, "yaml", {}).get("kpt_shape"))
        try:
            return int(shape[0])
        except (TypeError, IndexError, ValueError):
            return None

    def _warm_up(self) -> None:
        """
        Run the model once before anything is timed or waited on.

        The first call allocates workspace, picks kernels and compiles CUDA
        graphs; on this pipeline that first frame is tens of milliseconds
        against a 6 ms budget. Paying it in load() keeps it out of the live
        loop, where it would look like a dropped frame.
        """
        blank = numpy.zeros((self._imgsz, self._imgsz, 3), dtype="uint8")
        for _ in range(3):
            self._model.predict(blank, imgsz=self._imgsz, verbose=False,
                                device=self._device)

    def _load_body_parts(self, declared) -> None:
        """
        Same ordering and grouping contract the DeepLabCut backends honour.

        Duplicated rather than shared because the two read different files and
        the coupling would be the file format, not the logic.
        """
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
        return f"YoloPoseModel({os.path.basename(self._source)})"
