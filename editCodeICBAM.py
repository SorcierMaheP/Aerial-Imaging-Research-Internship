import importlib.util
import os

spec = importlib.util.find_spec("ultralytics")
path_init = spec.origin
package_path = os.path.dirname(path_init)
if spec is None or (not os.path.exists(package_path)):
    print("Ultralytics package was not found, run 'pip install ultralytics'")
    exit()


yolo_model_cfg = """# Ultralytics YOLO 🚀, AGPL-3.0 license
# YOLOv8 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect

# This is a modified version with modified attention modules

# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024] # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
  s: [0.33, 0.50, 1024] # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
  m: [0.67, 0.75, 768] # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
  l: [1.00, 1.00, 512] # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
  x: [1.00, 1.25, 512] # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs

# YOLOv8.0n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9

# YOLOv8.0n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 3, C2f, [512]] # 12
  #Improved CBAM Block
  - [-1, 1, ICBAM, [512,]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 3, C2f, [256]] # 15 (P3/8-small)
  #Improved CBAM Block
  - [-1, 1, ICBAM, [256,]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 3, C2f, [512]] # 18 (P4/16-medium)
  # Improved CBAM Block
  - [-1, 1, ICBAM, [512,]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]] # cat head P5
  - [-1, 3, C2f, [1024]] # 21 (P5/32-large)
  # Improved CBAM Block
  - [-1, 1, ICBAM, [1024,]]

  - [[17, 21, 25], 1, Detect, [nc]] # Detect(P3, P4, P5)
"""
model_cfg_path_rel = "cfg/models/v8"

model_cfg_path = os.path.join(package_path, model_cfg_path_rel)
if not os.path.exists(model_cfg_path):
    print(f"{model_cfg_path} does not exist. Could not write model config to file")
    exit()
config_file_path = os.path.join(model_cfg_path, "yolov8_ICBAM.yaml")
with open(config_file_path, "w+") as f:
    f.write(yolo_model_cfg)
print(f"Wrote to {config_file_path}")

# ----------------------------------------------------------------------------
ICBAMpython = '''# ICBAM block: Improved CBAM 

class ImprovedChannelAttention(nn.Module):

    def __init__(self, channels: int, k_size=3) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.maxPool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        yAvgPool = self.avgPool(x)
        yMaxPool = self.maxPool(x)

        avgConv = (
            self.fc(yAvgPool.squeeze(-1).transpose(-1, -2))
            .transpose(-1, -2)
            .unsqueeze(-1)
        )

        maxConv = self.fc(
            (yMaxPool.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        )

        # Multi-scale information fusion
        y = self.act(avgConv + maxConv)

        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(
            self.cv1(
                torch.cat(
                    [torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]],
                    1,
                )
            )
        )


class ICBAM(nn.Module):
    """Improved Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ImprovedChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))

'''

nn_modules_blocks = os.path.join(package_path, "nn/modules/block.py")
if not os.path.isfile(nn_modules_blocks):
    print(f"{nn_modules_blocks} does not exist. Could not write module")
    exit()

with open(nn_modules_blocks, "r") as f:
    contents = f.readlines()

index = 0
for n, val in enumerate(contents):
    if val.startswith("class"):
        index = n
        break
contents.insert(index, ICBAMpython)

index = 0
for n, val in enumerate(contents):
    if val.startswith("__all__"):
        index = n + 1
        break
contents.insert(index, '"ICBAM",\n')

with open(nn_modules_blocks, "w") as f:
    contents = "".join(contents)
    f.write(contents)
print(f"Wrote to {nn_modules_blocks}")

# ----------------------------------------------------------------------------
nn_modules_init = os.path.join(package_path, "nn/modules/__init__.py")
if not os.path.isfile(nn_modules_init):
    print(f"{nn_modules_init} does not exist. Could not write import")
    exit()

with open(nn_modules_init, "r") as f:
    contents = f.readlines()

index = 0
for n, val in enumerate(contents):
    if val.startswith("from .block import"):
        index = n + 1
        break
contents.insert(index, "ICBAM,\n")

index = 0
for n, val in enumerate(contents):
    if val.startswith("__all__"):
        index = n + 1
        break
contents.insert(index, '"ICBAM",\n')
with open(nn_modules_init, "w") as f:
    contents = "".join(contents)
    f.write(contents)

print(f"Wrote to {nn_modules_init}")

# ----------------------------------------------------------------------------

nn_tasks = os.path.join(package_path, "nn/tasks.py")
if not os.path.isfile(nn_tasks):
    print(f"{nn_tasks} does not exist. Could not write import")
    exit()

with open(nn_tasks, "r") as f:
    contents = f.readlines()

index = 0
for n, val in enumerate(contents):
    if val.startswith("from ultralytics.nn.modules"):
        index = n + 1
        break
contents.insert(index, "ICBAM,\n")

with open(nn_tasks, "w") as f:
    contents = "".join(contents)
    f.write(contents)
print(f"Wrote to {nn_tasks}")
