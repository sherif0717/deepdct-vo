# Track-B implementation components still located in the shared package

The following modules remain under `deepdct/` because the current shared
training/evaluation implementation imports them:

- `deepdct/training/motion_alignment.py`
- `deepdct/training/rotation_geometry.py`

The Track-A branch may later remove their hooks from the reproduction path.

Do not physically relocate these modules until the shared package has been
split or imports have been refactored.
