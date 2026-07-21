# Autotrainer - Inference
The inference module implements pose inference and any other low-level machine learning elements. The primary purpose is
to provide an implementation-agnostic interface to inference, such as the current dependency on DeepLabCut.

The module has three fundamental responsibilities:

* Manage pose inference running as an independent process from video acquisition and any post-inference analysis.
* Provide the ability to process from two independent frame queues (one at time, toggleable)
* Provide a consistent interface to the inference engine, such as DeepLabCut, that is agnostic to the implementation
  and allows for easy swapping of inference engines.

## Live Runtime Requirements

The current reachAQ DeepLabCut live path requires TensorFlow GPU acceleration
on a CUDA-capable NVIDIA GPU. It intentionally refuses CPU fallback because the
live acquisition rate cannot be supported reliably by CPU inference.

Before acquisition hardware starts, reachAQ checks the active NVIDIA kernel
driver with `nvidia-smi` and then validates that TensorFlow reports a GPU. A
Linux system using `nouveau`, a system without `nvidia-smi`, or a TensorFlow
runtime with no GPU fails the preflight before cameras are opened. The
application-level `--live-inference` and `--no-live-inference` flags can override
the saved setting for one run.

The preflight can be inspected directly from the configured environment:

```bash
python - <<'PY'
from autotrainer.inference import detect_gpu_runtime
print(detect_gpu_runtime(required_backend="tensorflow"))
PY
```


### TODO
* Allow explicit control of the frames processing per second.
  * Generally, the live processing can produce accurate results at something value like 30 fps.  Throttling the processing to this value may free resources for other parts of the application
* Expose more control of the in-memory "model" so that it can be used for testing.
  *  For testing and debugging it would be useful to be able to toggle whether the pose model is returning hits for each of the body parts, and at what location
