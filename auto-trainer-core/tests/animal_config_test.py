import pytest

from autotrainer.core import AnimalSubject


def test_save_load(tmp_path):
    animal = AnimalSubject(
        name="animal1",
        is_pellet_dcs=True,
        pellet_x=-1,
        pellet_y=1,
        pellet_z=2,
        target_y_limit=0.88,
    )
    assert isinstance(animal.id, str) and len(animal.id) > 0
    dest = tmp_path.joinpath("animal.json")
    animal.to_file(dest)
    animal2 = AnimalSubject.from_file(dest)
    assert animal == animal2
    # also ensure for change to any attribute:
    for k, v in animal.__dict__.items():
        if k == "version":  # version special, it's forced on write to file
            continue
        if isinstance(v, bool):
            v = not v
        elif isinstance(v, (int, float)):
            v += 1
        elif isinstance(v, str):
            v += "1"
        else:
            continue
        setattr(animal, k, v)
        assert animal != animal2
        animal.to_file(dest)
        animal2 = AnimalSubject.from_file(dest)
        assert animal == animal2, f"attribute {k} is not correctly saved/loaded"
