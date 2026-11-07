from autotrainer.video import VideoManager

print("Random Image Generator")
print("\tCamera 0: random://0")

flir_cameras = VideoManager.list_spin_cameras()

print("FLIR/Spinnaker")

if len(flir_cameras) == 0:
    print("\tNo cameras")
else:
    for i, sn in enumerate(flir_cameras):
        print(f"\tCamera {i}: spinnaker://{sn}")

print("File Playback")
print("\tCamera X: playback:///<path_to_file>")

