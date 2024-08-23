import importlib.util
import os

spec = importlib.util.find_spec("ultralytics")
path_init = spec.origin
package_path = os.path.dirname(path_init)
if spec is None or (not os.path.exists(package_path)):
    print("Ultralytics package was not found, run 'pip install ultralytics'")
    exit()


yolo_model_cfg = '''# Ultralytics YOLO , AGPL-3.0 license
# YOLOv10 ICBAM object detection model. For Usage examples see https://docs.ultralytics.com/tasks/detect

# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov10n.yaml' will call yolov10.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024]
  s: [0.33, 0.50, 1024]
  m: [0.67, 0.75, 768]
  l: [1.00, 1.00, 512]
  b: [0.67, 1.00, 512]
  x: [1.00, 1.25, 512]

backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, SCDown, [512, 3, 2]] # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, SCDown, [1024, 3, 2]] # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9
  - [-1, 1, PSA, [1024]] # 10

# YOLOv10.0n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 3, C2f, [512]] # 13

  #Improved CBAM Block
  - [-1, 1, ICBAM, [512,]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 3, C2f, [256]] # 16 (P3/8-small)

  #Improved CBAM Block
  - [-1, 1, ICBAM, [512,]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 3, C2f, [512]] # 19 (P4/16-medium)

  #Improved CBAM Block
  - [-1, 1, ICBAM, [512,]]

  - [-1, 1, SCDown, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 3, C2fCIB, [1024, True, True]] # 22 (P5/32-large)

  #Improved CBAM Block
  - [-1, 1, ICBAM, [512,]]

  - [[16, 19, 22], 1, v10Detect, [nc]] # Detect(P3, P4, P5)

'''
model_cfg_path_rel = 'cfg/models/v10'

model_cfg_path = os.path.join(package_path, model_cfg_path_rel)
if not os.path.exists(model_cfg_path):
    print(f'{model_cfg_path} does not exist. Could not write model config to file')
    exit()
config_file_path = os.path.join(model_cfg_path, 'yolov10_ICBAM.yaml')
with open(config_file_path, 'w+') as f:
    f.write(yolo_model_cfg)
print(f"Wrote to {config_file_path}")