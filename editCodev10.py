import importlib.util
import os

spec = importlib.util.find_spec("ultralytics")
path_init = spec.origin
package_path = os.path.dirname(path_init)
if spec is None or (not os.path.exists(package_path)):
    print("Ultralytics package was not found, run 'pip install ultralytics'")
    exit()


yolo_model_cfg = '''# Ultralytics YOLO , AGPL-3.0 license
# YOLOv10 HyP-ECA object detection model. For Usage examples see https://docs.ultralytics.com/tasks/detect

# Parameters
nc: 2
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

  #HyP-ECA block
  - [-1, 1, Improved_ECA, [512,]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 3, C2f, [256]] # 16 (P3/8-small)

  #Hyp-ECA Block
  - [-1, 1, Improved_ECA, [512,]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 3, C2f, [512]] # 19 (P4/16-medium)

  #Hyp-ECA Block
  - [-1, 1, Improved_ECA, [512,]]

  - [-1, 1, SCDown, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 3, C2fCIB, [1024, True, True]] # 22 (P5/32-large)

  #Hyp-ECA Block
  - [-1, 1, Improved_ECA, [512,]]

  - [[18, 22, 26], 1, v10Detect, [nc]] # Detect(P3, P4, P5)
'''
model_cfg_path_rel = 'cfg/models/v10'

model_cfg_path = os.path.join(package_path, model_cfg_path_rel)
if not os.path.exists(model_cfg_path):
    print(f'{model_cfg_path} does not exist. Could not write model config to file')
    exit()
config_file_path = os.path.join(model_cfg_path, 'yolov10_Improved_ECA.yaml')
with open(config_file_path, 'w+') as f:
    f.write(yolo_model_cfg)
print(f"Wrote to {config_file_path}")

# ----------------------------------------------------------------------------
Improved_ECA_python = '''# Anish Natekar code ECA block for yolo model
### Import necessary dependencies
class Improved_ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
    """

    def __init__(self, channel, k_size=3):
        super(Improved_ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)         # do maxpool instance too
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        y1 = self.max_pool(x)
        # Two different branches of ECA module
        avg_conv = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        max_conv = self.conv2(y1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = torch.sigmoid(avg_conv+max_conv)

        return x * y.expand_as(x)
'''

nn_modules_blocks = os.path.join(package_path, 'nn/modules/block.py')
if not os.path.isfile(nn_modules_blocks):
    print(f"{nn_modules_blocks} does not exist. Could not write module")
    exit()

with open(nn_modules_blocks, "r") as f:
    contents = f.readlines()
# for n, val in enumerate(contents[:20]):
#   print(n, val)
index = 0
for n, val in enumerate(contents):
    if val.startswith('class'):
        index = n
        break
contents.insert(index, Improved_ECA_python)

index = 0
for n, val in enumerate(contents):
    if val.startswith('__all__'):
        index = n+1
        break
contents.insert(index, '"Improved_ECA",\n')
with open(nn_modules_blocks, "w") as f:
    contents = "".join(contents)
    f.write(contents)
print(f"Wrote to {nn_modules_blocks}")

# ----------------------------------------------------------------------------
nn_modules_init = os.path.join(package_path, 'nn/modules/__init__.py')
if not os.path.isfile(nn_modules_init):
    print(f"{nn_modules_init} does not exist. Could not write import")
    exit()

with open(nn_modules_init, "r") as f:
    contents = f.readlines()
index = 0
for n, val in enumerate(contents):
    if val.startswith('from .block import'):
        index = n+1
        break
contents.insert(index, 'Improved_ECA,\n')

index = 0
for n, val in enumerate(contents):
    if val.startswith('__all__'):
        index = n+1
        break
contents.insert(index, '"Improved_ECA",\n')
with open(nn_modules_init, "w") as f:
    contents = "".join(contents)
    f.write(contents)
print(f"Wrote to {nn_modules_init}")
# ----------------------------------------------------------------------------
nn_tasks = os.path.join(package_path, 'nn/tasks.py')
if not os.path.isfile(nn_tasks):
    print(f"{nn_tasks} does not exist. Could not write import")
    exit()

with open(nn_tasks, "r") as f:
    contents = f.readlines()
index = 0
for n, val in enumerate(contents):
    if val.startswith('from ultralytics.nn.modules'):
        index = n+1
        break
contents.insert(index, 'Improved_ECA,\n')

with open(nn_tasks, "w") as f:
    contents = "".join(contents)
    f.write(contents)
print(f"Wrote to {nn_tasks}")
