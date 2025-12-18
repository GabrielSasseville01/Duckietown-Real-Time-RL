"""Constants."""

import yaml

camera_info_yaml = """
camera_matrix:
  cols: 3
  data:
  - 294.53932068591484
  - 0.0
  - 309.40712721751646
  - 0.0
  - 296.5367154664796
  - 228.72814869651435
  - 0.0
  - 0.0
  - 1.0
  rows: 3
camera_name: nyrobot
distortion_coefficients:
  cols: 5
  data:
  - -0.22642034632167934
  - 0.032424830545866784
  - -0.0030997885368560392
  - 0.00026050478624311846
  - 0.0
  rows: 1
distortion_model: plumb_bob
image_height: 480
image_width: 640
projection_matrix:
  cols: 4
  data:
  - 176.9991912841797
  - 0.0
  - 310.20938650828066
  - 0.0
  - 0.0
  - 178.88101196289062
  - 221.8908659114204
  - 0.0
  - 0.0
  - 0.0
  - 1.0
  - 0.0
  rows: 3
rectification_matrix:
  cols: 3
  data:
  - 1.0
  - 0.0
  - 0.0
  - 0.0
  - 1.0
  - 0.0
  - 0.0
  - 0.0
  - 1.0
  rows: 3
"""
camera_info_dict = yaml.safe_load(camera_info_yaml)
