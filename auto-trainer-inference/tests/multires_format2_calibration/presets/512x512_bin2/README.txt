512x512_bin2: one preset of a reachAQ multi-resolution 3D calibration.
Binning 2, 512x512 px, ROI offset (104, 12) in binned px.
Every pixel value here is in this preset's pixels: it applies only to video recorded
at this binning and ROI.

DeepLabCut 3D: camera_matrix/ is a DLC camera_matrix folder. Copy it to the 3D
project's camera_matrix folder. stereo_params.pickle keys the pair as "left-right",
so the project's config.yaml must set camera_names: ["left", "right"].

anipose: calibration.toml, in aniposelib's CameraGroup format, with the cameras'
poses in the motor frame and translations in mm.
