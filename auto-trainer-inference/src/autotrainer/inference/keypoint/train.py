"""Train a KeypointNet from labelled frames and write a loadable model directory.

    python -m autotrainer.inference.keypoint.train DATASET OUTPUT [--epochs N] ...

DATASET is a directory of square frames and their labels in the layout the
existing pose dataset was exported in:

    images/{train,val}/<name>.png
    labels/{train,val}/<name>.txt      one line: class, box (4), then x y v per part,
                                       x and y as fractions of the frame, v 0 = unlabelled

and a names file passed with --bodyparts, one name per line in label order (or
the yolo_pose.yaml of a model trained on the same data). OUTPUT receives
weights.pt (a plain state_dict) and keypoint_pose.yaml, which is everything
KeypointPoseModel needs.

In the app, training goes through train_arrays() with frames from a
keypoint.dataset.PoseDataset, which reports progress per epoch and can be
stopped and resumed exactly from its checkpoint (see train_arrays).

An unlabelled part is trained as absent: its heatmap target is empty. That is
what lets the confidence fall for a part that is not in view, and it is the
same treatment DeepLabCut gives a missing label. The cost is that a part an
annotator could see but did not label is taught as absent too.

The whole dataset is a few hundred small frames, so it is held on the GPU and
augmented there: one affine warp and a photometric jitter per frame per step,
with the labels moved by the same matrix. No horizontal flip - the cameras sit
either side of the animal, so a mirrored frame is a view the rig never sees,
and flipping would also have to swap every left and right part.

The final epoch is what gets written, not the best validation epoch. The held
out set is a few dozen frames, and picking a checkpoint on it would tune to
those frames rather than measure on them.
"""

import argparse
import json
import math
import os
import time
import typing

import numpy
import torch
import yaml

from .network import STRIDE, KeypointNet, decode

SIDECAR = "keypoint_pose.yaml"
WEIGHTS = "weights.pt"
CHECKPOINT = "checkpoint.pt"
HISTORY = "history.json"


# -- data -----------------------------------------------------------------

def _read_image(path: str) -> numpy.ndarray:
    import cv2
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f"cannot read {path!r}")
    if image.ndim == 3:
        image = image[..., :3].mean(axis=2)
    return image.astype("uint8")


def load_split(dataset: str, split: str, keypoints: int
               ) -> typing.Tuple[numpy.ndarray, numpy.ndarray, typing.List[str]]:
    """(N, S, S) uint8 frames, (N, K, 2) pixel labels with NaN where unlabelled, names."""
    image_dir = os.path.join(dataset, "images", split)
    label_dir = os.path.join(dataset, "labels", split)
    names = sorted(n for n in os.listdir(image_dir) if n.lower().endswith(".png"))
    frames, labels = [], []
    for name in names:
        image = _read_image(os.path.join(image_dir, name))
        rows, cols = image.shape
        if rows != cols:
            raise ValueError(f"{name}: {rows}x{cols} is not square")
        stem = os.path.splitext(name)[0]
        with open(os.path.join(label_dir, stem + ".txt"), encoding="utf-8") as handle:
            fields = handle.read().split()
        values = numpy.asarray(fields[5:], dtype="float64").reshape(-1, 3)
        if len(values) != keypoints:
            raise ValueError(f"{stem}: {len(values)} parts, expected {keypoints}")
        points = values[:, :2] * cols
        points[values[:, 2] <= 0] = numpy.nan
        frames.append(image)
        labels.append(points)
    return numpy.stack(frames), numpy.stack(labels).astype("float32"), names


def read_bodyparts(path: str) -> typing.List[str]:
    if path.endswith((".yaml", ".yml")):
        with open(path, encoding="utf-8") as handle:
            declared = (yaml.safe_load(handle) or {})["bodyparts"]
        if isinstance(declared, dict):
            return [part for parts in declared.values() for part in parts]
        return list(declared)
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


# -- augmentation ---------------------------------------------------------

def _augment(images: torch.Tensor, points: torch.Tensor, generator: torch.Generator,
             rotate: float, scale: float, shift: float):
    """
    Warp each frame by a random similarity and jitter its intensity.

    images (B, 1, S, S) in 0..1, points (B, K, 2) pixels (NaN unlabelled).
    The warp is built in pixel space and converted to grid_sample's
    normalised coordinates, so the labels move by exactly the same map.
    """
    b, _, size, _ = images.shape
    device = images.device

    def uniform(low, high):
        return torch.rand(b, generator=generator, device=device) * (high - low) + low

    angle = uniform(-rotate, rotate) * math.pi / 180
    zoom = torch.exp(uniform(-math.log(scale), math.log(scale)))
    tx, ty = uniform(-shift, shift) * size, uniform(-shift, shift) * size
    centre = (size - 1) / 2
    cos, sin = torch.cos(angle) * zoom, torch.sin(angle) * zoom
    # Forward map, pixel coordinates: p' = A (p - c) + c + t
    forward = torch.zeros(b, 3, 3, device=device)
    forward[:, 0, 0], forward[:, 0, 1] = cos, -sin
    forward[:, 1, 0], forward[:, 1, 1] = sin, cos
    forward[:, 0, 2] = centre + tx - (cos * centre - sin * centre)
    forward[:, 1, 2] = centre + ty - (sin * centre + cos * centre)
    forward[:, 2, 2] = 1
    # grid_sample asks, for each output pixel, where to read in the input.
    # With align_corners=True, normalised u = (p - c) / c, so the forward map
    # is u' = A u + t / c and its inverse u = A^-1 (u' - t / c), where A^-1 is
    # the opposite rotation over the same zoom. Written out rather than
    # inverted numerically: it is exact, and batched linalg.inv ran the GPU out
    # of memory a few hundred steps in on the Windows cu130 build.
    inv_cos, inv_sin = torch.cos(angle) / zoom, torch.sin(angle) / zoom
    theta = torch.zeros(b, 2, 3, device=device)
    theta[:, 0, 0], theta[:, 0, 1] = inv_cos, inv_sin
    theta[:, 1, 0], theta[:, 1, 1] = -inv_sin, inv_cos
    theta[:, 0, 2] = -(inv_cos * tx + inv_sin * ty) / centre
    theta[:, 1, 2] = -(-inv_sin * tx + inv_cos * ty) / centre
    grid = torch.nn.functional.affine_grid(theta, images.shape, align_corners=True)
    warped = torch.nn.functional.grid_sample(images, grid, mode="bilinear",
                                             padding_mode="zeros", align_corners=True)

    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=2)
    moved = (homogeneous @ forward.transpose(1, 2))[..., :2]

    gain = uniform(0.6, 1.5).view(b, 1, 1, 1)
    bias = uniform(-0.1, 0.1).view(b, 1, 1, 1)
    gamma = torch.exp(uniform(-0.4, 0.4)).view(b, 1, 1, 1)
    warped = (warped.clamp(0, 1) ** gamma) * gain + bias
    noise = uniform(0, 0.04).view(b, 1, 1, 1)
    warped = warped + noise * torch.randn(warped.shape, generator=generator, device=device)
    return warped.clamp(0, 1), moved


# -- targets and loss -----------------------------------------------------

def _targets(points: torch.Tensor, cells: int):
    """
    Positive cell index, offsets and validity per part.

    A part counts as labelled here only if it is labelled and still inside the
    frame after augmentation; a part warped off the edge is genuinely absent
    from that view, so it is trained as absent like any unlabelled one.
    """
    u = (points + 0.5) / STRIDE                       # continuous cell coordinate
    valid = torch.isfinite(u).all(dim=2) & (u >= 0).all(dim=2) & (u < cells).all(dim=2)
    u = torch.where(valid.unsqueeze(2), u, torch.zeros_like(u))
    cell = u.floor().clamp(0, cells - 1)
    return u, cell, valid


def loss_terms(output: torch.Tensor, points: torch.Tensor, sigma: float = 1.0,
               alpha: float = 2.0, beta: float = 4.0):
    """
    Penalty-reduced focal loss on the heatmaps, L1 on the offsets.

    The heatmap target is a Gaussian around the label with the label's own
    cell set to exactly one; cells near a positive are penalised less for
    firing, so the peak is not pushed to be one-cell sharp. Offsets are
    supervised over the 3x3 cells around the label, so the decode still lands
    on the label when the peak comes out one cell off.
    """
    n, channels, cells, _ = output.shape
    k = channels // 3
    logits = output[:, :k].float()
    u, cell, valid = _targets(points, cells)

    grid = torch.arange(cells, device=output.device, dtype=torch.float32) + 0.5
    dx = grid.view(1, 1, 1, cells) - u[..., 0].view(n, k, 1, 1)
    dy = grid.view(1, 1, cells, 1) - u[..., 1].view(n, k, 1, 1)
    gauss = torch.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))
    gauss = gauss * valid.view(n, k, 1, 1)
    col = torch.arange(cells, device=output.device).view(1, 1, 1, cells)
    row = torch.arange(cells, device=output.device).view(1, 1, cells, 1)
    positive = ((col == cell[..., 0].long().view(n, k, 1, 1))
                & (row == cell[..., 1].long().view(n, k, 1, 1))
                & valid.view(n, k, 1, 1))

    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    log_p = torch.nn.functional.logsigmoid(logits)
    log_not_p = torch.nn.functional.logsigmoid(-logits)
    pos_loss = -(((1 - p) ** alpha) * log_p)[positive].sum()
    neg_loss = -(((1 - gauss) ** beta) * (p ** alpha) * log_not_p)[~positive].sum()
    count = positive.sum().clamp(min=1)
    heat = (pos_loss + neg_loss) / count

    near = ((col - cell[..., 0].long().view(n, k, 1, 1)).abs() <= 1) \
        & ((row - cell[..., 1].long().view(n, k, 1, 1)).abs() <= 1) \
        & valid.view(n, k, 1, 1)
    offsets = output[:, k:].float().reshape(n, 2, k, cells, cells)
    target_x = u[..., 0].view(n, k, 1, 1) - grid.view(1, 1, 1, cells)
    target_y = u[..., 1].view(n, k, 1, 1) - grid.view(1, 1, cells, 1)
    error = (offsets[:, 0] - target_x).abs() + (offsets[:, 1] - target_y).abs()
    offset = error[near].sum() / near.sum().clamp(min=1)
    return heat, offset


# -- evaluation -----------------------------------------------------------

@torch.no_grad()
def predict_frames(net: KeypointNet, frames: numpy.ndarray, device: str,
                   batch: int = 64) -> numpy.ndarray:
    net.eval()
    out = []
    for start in range(0, len(frames), batch):
        chunk = torch.from_numpy(frames[start:start + batch]).to(device)
        chunk = chunk.float().div_(255).unsqueeze(1)
        out.append(decode(net(chunk)).cpu().numpy())
    return numpy.concatenate(out)


def summarise(predicted: numpy.ndarray, labels: numpy.ndarray, gate: float = 0.1) -> dict:
    """Median error and detection rate over labelled parts, for the training log."""
    labelled = numpy.isfinite(labels).all(axis=2)
    error = numpy.linalg.norm(predicted[..., :2] - labels, axis=2)[labelled]
    score = predicted[..., 2][labelled]
    return {"median_px": float(numpy.median(error)),
            "detected": float((score >= gate).mean()),
            "median_px_detected": float(numpy.median(error[score >= gate]))
            if (score >= gate).any() else float("nan")}


# -- training -------------------------------------------------------------

def autocast_dtype(device: str) -> typing.Optional[torch.dtype]:
    """The reduced precision training runs in on this device, or None for fp32.

    bf16 needs compute capability 8.0. Before that torch emulates it, and on
    the Quadro T1000 (7.5, full network, batch 32) a step took 489 ms in bf16,
    417 in fp16 and 126 in fp32, so older cards train in fp32.
    """
    if not device.startswith("cuda"):
        return None
    if torch.cuda.get_device_capability(device)[0] < 8:
        return None
    return torch.bfloat16


def _atomic_save(obj, path: str) -> None:
    temporary = path + ".tmp"
    torch.save(obj, temporary)
    os.replace(temporary, path)


def _snapshot(net, optimiser, scheduler, generator, epoch, history, frames, epochs) -> dict:
    """Everything needed to continue from the start of `epoch`, copied to the CPU."""
    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().to("cpu", copy=True)
        if isinstance(value, dict):
            return {key: cpu(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(cpu(item) for item in value)
        return value
    return {"epoch": epoch, "epochs": epochs, "frames": frames,
            "net": cpu(net.state_dict()), "optimiser": cpu(optimiser.state_dict()),
            "scheduler": scheduler.state_dict(), "generator": generator.get_state(),
            "history": [dict(line) for line in history]}


def train_arrays(train_frames: numpy.ndarray, train_points: numpy.ndarray,
                 val_frames: numpy.ndarray, val_points: numpy.ndarray,
                 bodyparts: typing.Sequence[str], output: str, *, epochs: int = 300,
                 batch: int = 32, lr: float = 2e-3, weight_decay: float = 1e-4,
                 seed: int = 0, device: str = "cuda", widths=(32, 64, 128, 192),
                 blocks=(1, 2, 2, 2), neck: int = 64, rotate: float = 25.0,
                 scale: float = 1.25, shift: float = 0.1, sigma: float = 1.0,
                 progress: typing.Optional[typing.Callable[[int, int, dict], None]] = None,
                 should_stop: typing.Optional[typing.Callable[[], bool]] = None,
                 checkpoint_every: int = 10, resume: bool = False,
                 provenance: typing.Optional[dict] = None, log=print) -> dict:
    """
    Train on (N, S, S) uint8 frames with (N, K, 2) pixel labels (NaN absent).

    progress(epoch, epochs, line) is called after every epoch. should_stop()
    is polled every step; when it answers True the run writes CHECKPOINT,
    holding the state at the start of the epoch in progress, and returns
    {"status": "stopped"}. The partial epoch is discarded, so a resumed run
    (resume=True) replays it from the same generator state and ends exactly
    where an uninterrupted run would. A checkpoint is also written every
    checkpoint_every epochs, so a killed process loses little.
    """
    k = len(bodyparts)
    if len(train_frames) == 0:
        raise ValueError("there are no labelled training frames")
    size = train_frames.shape[1]
    if size % 32:
        raise ValueError(f"frame size {size} must be a multiple of 32")
    has_val = len(val_frames) > 0
    os.makedirs(output, exist_ok=True)
    checkpoint_path = os.path.join(output, CHECKPOINT)

    torch.manual_seed(seed)
    numpy.random.seed(seed)
    images = torch.from_numpy(train_frames).to(device).float().div_(255).unsqueeze(1)
    points = torch.from_numpy(train_points.astype("float32")).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    net = KeypointNet(k, widths=widths, blocks=blocks, neck=neck).to(device)
    optimiser = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, math.ceil(len(images) / batch))
    total = epochs * steps_per_epoch
    warmup = min(500, total // 10)

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule)
    dtype = autocast_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    start_epoch, history = 0, []
    if resume and os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if state["frames"] != len(images) or state["epochs"] != epochs:
            raise ValueError(
                f"the checkpoint was for {state['frames']} frames and {state['epochs']} "
                f"epochs, not {len(images)} and {epochs}; start a new run instead")
        net.load_state_dict(state["net"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        generator.set_state(state["generator"])
        start_epoch, history = state["epoch"], list(state["history"])
        log(f"resuming at epoch {start_epoch + 1} of {epochs}")

    log(f"training {k} parts on {len(images)} frames of {size}px, "
        f"{epochs} epochs x {steps_per_epoch} steps")
    started = time.time()
    report_every = max(1, epochs // 10)
    for epoch in range(start_epoch, epochs):
        snapshot = _snapshot(net, optimiser, scheduler, generator, epoch, history,
                             len(images), epochs)
        if epoch > start_epoch and epoch % checkpoint_every == 0:
            _atomic_save(snapshot, checkpoint_path)
        net.train()
        order = torch.randperm(len(images), generator=generator, device=device)
        for start in range(0, len(images), batch):
            if should_stop is not None and should_stop():
                _atomic_save(snapshot, checkpoint_path)
                log(f"stopped; resumes at epoch {epoch + 1}")
                return {"status": "stopped", "epoch": epoch, "history": history,
                        "seconds": time.time() - started}
            index = order[start:start + batch]
            x, p = _augment(images[index], points[index], generator, rotate, scale, shift)
            with torch.autocast(device_type="cuda", dtype=dtype or torch.float32,
                                enabled=dtype is not None):
                out = net(x)
            heat, offset = loss_terms(out, p, sigma=sigma)
            loss = heat + offset
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
        line = {"epoch": epoch + 1, "heat": heat.item(), "offset": offset.item()}
        if has_val and ((epoch + 1) % report_every == 0 or epoch + 1 == epochs):
            line.update(summarise(predict_frames(net, val_frames, device), val_points))
            history.append(line)
            log(f"{line}  ({time.time() - started:.0f}s)")
        elif not has_val and epoch + 1 == epochs:
            history.append(line)
        if progress is not None:
            progress(epoch + 1, epochs, line)

    torch.save(net.state_dict(), os.path.join(output, WEIGHTS))
    sidecar = {"weights": WEIGHTS, "imgsz": size, "network": net.config(),
               "bodyparts": list(bodyparts)}
    sidecar["training"] = {**(provenance or {}), "epochs": epochs, "seed": seed,
                           "train_frames": int(len(train_frames)),
                           "val_frames": int(len(val_frames)),
                           "torch": str(torch.__version__)}
    with open(os.path.join(output, SIDECAR), "w", encoding="utf-8") as handle:
        yaml.safe_dump(sidecar, handle, sort_keys=False)
    with open(os.path.join(output, HISTORY), "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=1)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    return {"status": "finished", "epoch": epochs, "history": history,
            "seconds": time.time() - started}


def train(dataset: str, output: str, bodyparts: typing.List[str], epochs: int = 300,
          **kwargs) -> dict:
    """Train from an exported YOLO-layout directory (images/, labels/)."""
    k = len(bodyparts)
    train_frames, train_labels, _ = load_split(dataset, "train", k)
    if os.path.isdir(os.path.join(dataset, "images", "val")):
        val_frames, val_labels, _ = load_split(dataset, "val", k)
    else:
        val_frames = train_frames[:0]
        val_labels = train_labels[:0]
    return train_arrays(train_frames, train_labels, val_frames, val_labels, bodyparts,
                        output, epochs=epochs, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument("--bodyparts", required=True,
                        help="names file, one per line, or a yaml with a bodyparts key")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--widths", type=int, nargs=4, default=(32, 64, 128, 192))
    parser.add_argument("--blocks", type=int, nargs=4, default=(1, 2, 2, 2))
    parser.add_argument("--neck", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    train(args.dataset, args.output, read_bodyparts(args.bodyparts), epochs=args.epochs,
          batch=args.batch, lr=args.lr, seed=args.seed, device=args.device,
          widths=args.widths, blocks=args.blocks, neck=args.neck, sigma=args.sigma)


if __name__ == "__main__":
    main()
