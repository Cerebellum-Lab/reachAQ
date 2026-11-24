import logging
import sys

from reachaq.inference import PoseModel, MemoryPoseModel
from reachaq.pose_plugins.deeplabcut import DlcPoseModel

logging.getLogger("reachaq").setLevel(logging.DEBUG)


def show_model(m: PoseModel, name: str):
    m.load()

    print(f"=== Model: {name} ===")
    print(f"Body Part Categories:\t{m.body_part_categories}")
    print(f"Body Parts:\t\t\t\t{m.body_parts}")


if __name__ == '__main__':
    model = MemoryPoseModel()
    show_model(model, "Memory")

    for idx in range(1, len(sys.argv)):
        model = DlcPoseModel(sys.argv[idx])
        show_model(model, sys.argv[idx])
