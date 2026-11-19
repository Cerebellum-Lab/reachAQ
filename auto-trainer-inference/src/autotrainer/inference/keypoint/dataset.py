"""The labelled frames a keypoint model is trained on, as reachAQ stores them.

    <root>/<name>/
        dataset.yaml     version, bodyparts, imgsz, val_sessions, created, source
        frames/<session>/<stem>.png     FRAME_SIZE x FRAME_SIZE grey
        labels.csv       frame, session, status, <part>_x, <part>_y, ...

Every frame is stored at the size the live model runs at. KeypointPoseModel
resizes whatever square frame it is given to its imgsz, so a 1024 px recording
resized to 256 here is exactly what live inference sees, and the labels are in
the pixels of the stored frame.

A row is "pending" until the operator marks it done; only "labelled" rows are
trained or scored on. That is what lets a frame be pre-labelled by the current
model without its guesses becoming training data unreviewed.

The split is by session, never by frame. Both cameras of a recording see the
same reach, and consecutive frames of one camera are nearly identical, so a
frame-level split would score the model on frames it has effectively trained
on. A session is the recording: the camera suffix is stripped from the name.

Pixel centres are on integer coordinates, the convention keypoint.train
encodes its targets in, so a resize maps a coordinate as (p + 0.5) * s - 0.5.
"""

import csv
import dataclasses
import datetime
import hashlib
import io
import os
import re
import typing

import numpy
import yaml

FRAME_SIZE = 256
DATASET_FILE = "dataset.yaml"
LABELS_FILE = "labels.csv"
FRAMES_DIR = "frames"
VERSION = 1

PENDING = "pending"
LABELLED = "labelled"
STATUSES = (PENDING, LABELLED)

# Held-out share of sessions when the operator has not chosen.
DEFAULT_VAL_FRACTION = 0.1

_CAMERA_SUFFIX = re.compile(r"_(left|right|[A-Za-z]*[Cc]am(-\d+)?)$")


def session_of(name: str) -> str:
    """The recording a video or labelled-data folder belongs to."""
    return _CAMERA_SUFFIX.sub("", name)


def _atomic_write(path: str, data: bytes) -> None:
    temporary = path + ".tmp"
    with open(temporary, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _to_grey(image: numpy.ndarray) -> numpy.ndarray:
    if image.ndim == 3:
        image = image[..., :3].mean(axis=2)
    if image.dtype != numpy.uint8:
        # 16-bit PNGs: keep the top byte, as the camera's 8-bit output would.
        image = image / (257.0 if image.max() > 255 else 1.0)
    return image.round().clip(0, 255).astype("uint8")


def _resize(image: numpy.ndarray, name: str) -> typing.Tuple[numpy.ndarray, float]:
    """Square grey frame at FRAME_SIZE, and the scale applied to it."""
    import cv2
    rows, cols = image.shape[:2]
    if rows != cols:
        raise ValueError(f"{name}: {cols}x{rows} is not square; the live model "
                         f"only takes square frames")
    image = _to_grey(image)
    if rows == FRAME_SIZE:
        return image, 1.0
    scale = FRAME_SIZE / rows
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, (FRAME_SIZE, FRAME_SIZE), interpolation=interpolation), scale


def _rescale(points: numpy.ndarray, scale: float) -> numpy.ndarray:
    if scale == 1.0:
        return points
    return (points + 0.5) * scale - 0.5


@dataclasses.dataclass
class FrameRow:
    frame: str              # path under frames/, "/"-separated
    session: str
    status: str
    points: numpy.ndarray   # (K, 2) float32, NaN where the part is not visible


class PoseDataset:
    """A dataset directory. Edits are in memory until save()."""

    def __init__(self, path: str, bodyparts: typing.Sequence[str],
                 val_sessions: typing.Sequence[str] = (), meta: typing.Optional[dict] = None):
        self.path = path
        self.bodyparts = list(bodyparts)
        self.val_sessions = sorted(set(val_sessions))
        self.meta = dict(meta or {})
        self.rows: typing.List[FrameRow] = []

    # -- files ------------------------------------------------------------

    @property
    def name(self) -> str:
        return os.path.basename(os.path.normpath(self.path))

    @classmethod
    def create(cls, root: str, name: str, bodyparts: typing.Sequence[str],
               source: str = "") -> "PoseDataset":
        if not name or os.sep in name or "/" in name:
            raise ValueError(f"{name!r} is not a usable dataset name")
        if len(set(bodyparts)) != len(bodyparts) or not bodyparts:
            raise ValueError("body part names must be unique and non-empty")
        path = os.path.join(root, name)
        if os.path.exists(path):
            raise FileExistsError(f"dataset {path!r} already exists")
        os.makedirs(os.path.join(path, FRAMES_DIR))
        data = cls(path, bodyparts, meta={
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "source": source})
        data.save()
        return data

    @classmethod
    def open(cls, path: str) -> "PoseDataset":
        with open(os.path.join(path, DATASET_FILE), encoding="utf-8") as handle:
            spec = yaml.safe_load(handle) or {}
        if spec.get("version") != VERSION:
            raise ValueError(f"{path!r}: dataset version {spec.get('version')!r}, "
                             f"this build reads {VERSION}")
        if int(spec.get("imgsz", FRAME_SIZE)) != FRAME_SIZE:
            raise ValueError(f"{path!r}: frames are {spec.get('imgsz')} px, expected {FRAME_SIZE}")
        data = cls(path, spec["bodyparts"], spec.get("val_sessions") or (),
                   {key: spec[key] for key in ("created", "source") if key in spec})
        labels = os.path.join(path, LABELS_FILE)
        if os.path.exists(labels):
            with open(labels, encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader)
                expected = data._header()
                if header != expected:
                    raise ValueError(f"{labels!r}: columns do not match the body parts")
                k = len(data.bodyparts)
                for fields in reader:
                    values = [float(v) if v else numpy.nan for v in fields[3:]]
                    points = numpy.asarray(values, "float32").reshape(k, 2)
                    data.rows.append(FrameRow(fields[0], fields[1], fields[2], points))
        return data

    def _header(self) -> typing.List[str]:
        return ["frame", "session", "status"] + [
            f"{part}_{axis}" for part in self.bodyparts for axis in ("x", "y")]

    def save(self) -> None:
        spec = {"version": VERSION, "imgsz": FRAME_SIZE, "bodyparts": self.bodyparts,
                "val_sessions": self.val_sessions, **self.meta}
        _atomic_write(os.path.join(self.path, DATASET_FILE),
                      yaml.safe_dump(spec, sort_keys=False).encode("utf-8"))
        text = io.StringIO()
        writer = csv.writer(text, lineterminator="\n")
        writer.writerow(self._header())
        for row in self.rows:
            values = ["" if numpy.isnan(v) else f"{v:.3f}" for v in row.points.reshape(-1)]
            writer.writerow([row.frame, row.session, row.status, *values])
        _atomic_write(os.path.join(self.path, LABELS_FILE), text.getvalue().encode("utf-8"))

    # -- rows -------------------------------------------------------------

    def row(self, frame: str) -> FrameRow:
        for row in self.rows:
            if row.frame == frame:
                return row
        raise KeyError(frame)

    def image_path(self, frame: str) -> str:
        return os.path.join(self.path, FRAMES_DIR, *frame.split("/"))

    def image(self, frame: str) -> numpy.ndarray:
        import cv2
        image = cv2.imread(self.image_path(frame), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise OSError(f"cannot read frame {frame!r}")
        return image

    def add_image(self, image: numpy.ndarray, session: str, stem: str,
                  points: typing.Optional[numpy.ndarray] = None,
                  status: str = PENDING) -> str:
        """Store a FRAME_SIZE grey frame and add its row; returns the frame id."""
        import cv2
        if image.shape != (FRAME_SIZE, FRAME_SIZE):
            raise ValueError(f"frame must be {FRAME_SIZE}x{FRAME_SIZE} grey, got {image.shape}")
        frame = f"{session}/{stem}.png"
        if any(row.frame == frame for row in self.rows):
            raise ValueError(f"{frame!r} is already in the dataset")
        path = self.image_path(frame)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not cv2.imwrite(path, image):
            raise OSError(f"cannot write {path!r}")
        if points is None:
            points = numpy.full((len(self.bodyparts), 2), numpy.nan, "float32")
        self.rows.append(FrameRow(frame, session, status, self._checked(points)))
        return frame

    def _checked(self, points) -> numpy.ndarray:
        points = numpy.asarray(points, "float32")
        if points.shape != (len(self.bodyparts), 2):
            raise ValueError(f"points must be ({len(self.bodyparts)}, 2), got {points.shape}")
        return points.copy()

    def set_points(self, frame: str, points) -> None:
        self.row(frame).points = self._checked(points)

    def set_status(self, frame: str, status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        self.row(frame).status = status

    def remove(self, frame: str) -> None:
        self.rows.remove(self.row(frame))
        try:
            os.remove(self.image_path(frame))
        except FileNotFoundError:
            pass

    def frames_from(self, session: str) -> typing.List[str]:
        return [row.frame for row in self.rows if row.session == session]

    # -- split ------------------------------------------------------------

    def sessions(self) -> typing.List[str]:
        return sorted({row.session for row in self.rows})

    def set_val(self, session: str, held_out: bool) -> None:
        chosen = set(self.val_sessions)
        (chosen.add if held_out else chosen.discard)(session)
        self.val_sessions = sorted(chosen)

    def default_split(self, fraction: float = DEFAULT_VAL_FRACTION) -> None:
        """Hold out about `fraction` of the sessions, at least one when there
        are two or more. Chosen by a hash of the name, so the same sessions are
        picked on any machine and adding a session does not reshuffle the rest."""
        sessions = self.sessions()
        if len(sessions) < 2:
            self.val_sessions = []
            return
        count = max(1, round(fraction * len(sessions)))
        ranked = sorted(sessions, key=lambda s: hashlib.sha1(s.encode()).hexdigest())
        self.val_sessions = sorted(ranked[:count])

    def _selected(self, split: str, status: str) -> typing.List[FrameRow]:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        held = set(self.val_sessions)
        return [row for row in self.rows if row.status == status
                and (row.session in held) == (split == "val")]

    def arrays(self, split: str, status: str = LABELLED
               ) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
        """(N, S, S) uint8 frames and (N, K, 2) float32 points for one split."""
        rows = self._selected(split, status)
        k = len(self.bodyparts)
        if not rows:
            return (numpy.zeros((0, FRAME_SIZE, FRAME_SIZE), "uint8"),
                    numpy.zeros((0, k, 2), "float32"))
        return (numpy.stack([self.image(row.frame) for row in rows]),
                numpy.stack([row.points for row in rows]).astype("float32"))

    def counts(self) -> dict:
        labelled = [row for row in self.rows if row.status == LABELLED]
        held = set(self.val_sessions)
        return {"frames": len(self.rows),
                "labelled": len(labelled),
                "pending": len(self.rows) - len(labelled),
                "train": sum(row.session not in held for row in labelled),
                "val": sum(row.session in held for row in labelled),
                "sessions": len(self.sessions())}

    def content_hash(self) -> str:
        """sha256 of the body parts and every labelled row with its frame bytes.

        Pending rows, row order and the split do not change it: it identifies
        the labels a model was trained on. The split is recorded beside it.
        """
        digest = hashlib.sha256()
        digest.update("\n".join(self.bodyparts).encode())
        for row in sorted((r for r in self.rows if r.status == LABELLED),
                          key=lambda r: r.frame):
            values = ",".join("" if numpy.isnan(v) else f"{v:.3f}"
                              for v in row.points.reshape(-1))
            digest.update(f"\n{row.frame}|{row.session}|{values}|".encode())
            with open(self.image_path(row.frame), "rb") as handle:
                digest.update(hashlib.sha256(handle.read()).digest())
        return digest.hexdigest()

    # -- video ------------------------------------------------------------

    def add_frames(self, video: str, indices: typing.Iterable[int],
                   session: typing.Optional[str] = None) -> typing.List[str]:
        """Decode these frames of a recorded video and add them as pending.

        Nothing is stored unless every requested frame exists.
        """
        stem = os.path.splitext(os.path.basename(video))[0]
        session = session or session_of(stem)
        wanted = sorted(set(int(i) for i in indices))
        decoded = read_frames(video, wanted)
        missing = [i for i in wanted if i not in decoded]
        if missing:
            raise IndexError(f"{video!r} has no frame {missing[0]} "
                             f"({len(decoded)} of {len(wanted)} found)")
        added = []
        for index in wanted:
            image, _ = _resize(decoded[index], video)
            name = f"{stem}_{index:06d}"
            if any(row.frame == f"{session}/{name}.png" for row in self.rows):
                continue
            added.append(self.add_image(image, session, name))
        return added


# -- video reading ----------------------------------------------------------

def video_frame_count(video: str) -> int:
    """The container's frame count; an estimate for some encoders."""
    import cv2
    capture = cv2.VideoCapture(video)
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def read_frames(video: str, indices: typing.Iterable[int],
                size: typing.Optional[int] = None) -> typing.Dict[int, numpy.ndarray]:
    """Frames by index; with `size`, shrunk to size x size grey as they are read.

    Each frame is reached by seeking, which OpenCV's FFmpeg backend does
    exactly (it decodes forward from the previous keyframe): on christielab10's
    recordings every seek matched sequential decoding, at about 1 ms per frame,
    where decoding from the start would take minutes for a frame late in a long
    session. If the position after a read is not the frame asked for, the
    backend cannot seek exactly, and the rest is decoded sequentially instead.
    """
    import cv2
    wanted = sorted(set(int(i) for i in indices))
    found: typing.Dict[int, numpy.ndarray] = {}
    if not wanted:
        return found

    def keep(index, image):
        if size is not None:
            image = cv2.resize(_to_grey(image), (size, size), interpolation=cv2.INTER_AREA)
        found[index] = image

    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise OSError(f"cannot open video {video!r}")
    try:
        remaining = list(wanted)
        while remaining:
            index = remaining[0]
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if not ok:
                break
            if int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) != index + 1:
                break
            keep(index, image)
            remaining.pop(0)
        else:
            return found
    finally:
        capture.release()

    # Inexact seek, or a read past the end: decode the rest from the start.
    pending = set(remaining)
    capture = cv2.VideoCapture(video)
    try:
        index, last = 0, max(pending)
        while index <= last:
            ok, image = capture.read()
            if not ok:
                break
            if index in pending:
                keep(index, image)
            index += 1
    finally:
        capture.release()
    return found


def pick_diverse(video: str, n: int, exclude: typing.Iterable[int] = (),
                 samples: int = 1000, seed: int = 0) -> typing.List[int]:
    """n frame indices that look different from each other.

    Samples up to `samples` frames evenly across the video, shrinks them to
    32x32, clusters them with k-means and takes the sample nearest each
    centre, as DeepLabCut's kmeans extraction does.
    """
    total = video_frame_count(video)
    excluded = set(int(i) for i in exclude)
    if total <= 0 or n <= 0:
        return []
    step = max(1, total // samples)
    candidates = [i for i in range(0, total, step) if i not in excluded]
    decoded = read_frames(video, candidates, size=32)
    candidates = [i for i in candidates if i in decoded]
    if len(candidates) <= n:
        return candidates
    features = numpy.stack([decoded[i].reshape(-1) for i in candidates]).astype("float32") / 255
    labels = _kmeans(features, n, seed)
    picked = []
    for cluster in range(n):
        members = numpy.flatnonzero(labels == cluster)
        if not len(members):
            continue
        centre = features[members].mean(axis=0)
        nearest = members[numpy.argmin(((features[members] - centre) ** 2).sum(axis=1))]
        picked.append(candidates[nearest])
    # An emptied cluster leaves a gap; fill it with the farthest remaining frames.
    remaining = [i for i in range(len(candidates)) if candidates[i] not in picked]
    while len(picked) < n and remaining:
        chosen = numpy.stack([features[candidates.index(p)] for p in picked])
        distance = ((features[remaining][:, None] - chosen[None]) ** 2).sum(axis=2).min(axis=1)
        picked.append(candidates[remaining.pop(int(numpy.argmax(distance)))])
    return sorted(picked)


def _kmeans(features: numpy.ndarray, k: int, seed: int, iterations: int = 30) -> numpy.ndarray:
    rng = numpy.random.default_rng(seed)
    # k-means++ initialisation.
    centres = [features[rng.integers(len(features))]]
    for _ in range(1, k):
        distance = ((features[:, None] - numpy.stack(centres)[None]) ** 2).sum(axis=2).min(axis=1)
        total = distance.sum()
        probability = distance / total if total > 0 else None
        centres.append(features[rng.choice(len(features), p=probability)])
    centres = numpy.stack(centres)
    labels = numpy.zeros(len(features), int)
    for _ in range(iterations):
        distance = ((features[:, None] - centres[None]) ** 2).sum(axis=2)
        new = distance.argmin(axis=1)
        if (new == labels).all() and _ > 0:
            break
        labels = new
        for cluster in range(k):
            members = features[labels == cluster]
            if len(members):
                centres[cluster] = members.mean(axis=0)
    return labels


# -- DeepLabCut import ------------------------------------------------------

def _read_dlc_csv(path: str) -> typing.Tuple[typing.List[typing.Tuple[str, str, int]],
                                             typing.List[typing.List[str]]]:
    """((column, part, axis) per data column, rows of (image path, values...))."""
    with open(path, encoding="utf-8", newline="") as handle:
        table = list(csv.reader(handle))
    # A multi-animal file has an "individuals" row second, so it fails here too.
    if len(table) < 3 or table[0][:1] != ["scorer"] or table[1][:1] != ["bodyparts"]:
        raise ValueError(f"{path!r} is not a single-animal DeepLabCut CollectedData file")
    parts_row, coords_row = table[1], table[2]
    width = next(i for i in range(1, len(parts_row)) if parts_row[i])
    columns = [(j, parts_row[j], 0 if coords_row[j] == "x" else 1)
               for j in range(width, len(parts_row)) if coords_row[j] in ("x", "y")]
    rows = []
    for fields in table[3:]:
        if not fields:
            continue
        image = "/".join(fields[:width]).replace("\\", "/")
        rows.append([image] + fields)
    return columns, rows


def import_dlc(project_dir: str, root: str, name: str) -> PoseDataset:
    """Copy a DeepLabCut project's labelled frames into a new dataset.

    The project is only read. Frames with no labelled part are skipped: in a
    DeepLabCut project those are extracted frames nobody labelled.
    """
    import cv2
    with open(os.path.join(project_dir, "config.yaml"), encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if config.get("multianimalproject"):
        raise ValueError("multi-animal DeepLabCut projects are not supported")
    bodyparts = list(config.get("bodyparts") or ())
    labelled_dir = os.path.join(project_dir, "labeled-data")
    sources = []
    for folder in sorted(os.listdir(labelled_dir)):
        directory = os.path.join(labelled_dir, folder)
        if not os.path.isdir(directory):
            continue
        for file in sorted(os.listdir(directory)):
            if file.startswith("CollectedData_") and file.endswith(".csv"):
                sources.append((folder, os.path.join(directory, file)))

    # Read and check everything before writing anything.
    entries = []
    for folder, csv_path in sources:
        columns, rows = _read_dlc_csv(csv_path)
        for fields in rows:
            points = numpy.full((len(bodyparts), 2), numpy.nan, "float32")
            for column, part, axis in columns:
                if part in bodyparts and column < len(fields) - 1 and fields[column + 1]:
                    points[bodyparts.index(part), axis] = float(fields[column + 1])
            # A part needs both coordinates to count.
            points[numpy.isnan(points).any(axis=1)] = numpy.nan
            if numpy.isnan(points).all():
                continue
            image_path = os.path.join(project_dir, *fields[0].split("/"))
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise OSError(f"cannot read {image_path!r}")
            image, scale = _resize(image, image_path)
            stem = f"{folder}_{os.path.splitext(os.path.basename(image_path))[0]}"
            entries.append((image, session_of(folder), stem, _rescale(points, scale)))

    data = PoseDataset.create(root, name, bodyparts, source=os.path.abspath(project_dir))
    for image, session, stem, points in entries:
        data.add_image(image, session, stem, points, status=LABELLED)
    data.default_split()
    data.save()
    return data
