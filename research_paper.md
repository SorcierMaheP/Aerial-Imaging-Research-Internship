# Introduction
# Related Works
So far various attention mechanisms have been introduced to improve performance of CNNs. SE-Net created a method for extracting channel information. ECA-Net presented an efficient algorithm for channel attention with relatively good performance. CBAM mixed channel attention with spatial attention to enhanmce features in two separate dimensions.  HyP-ECA enhanced the results of ECA-Net and introduced it to YOLOv8 to improve detection of smaller objects.  

## CBAM
The CBAM attention process works as follows: First the input feature map is is enhanced by the channel attention block, the output of which is enhanced by the spatial attention block.
### Channel Attention block
The channel attention block squeezes the spatial dimension using both average pooling and max pooling simultaneously to generate two spatial context descriptors Fcavg and Fcmax. A shared multilayer then generates the channels attention maps Mc ∈ RC×1×1. 
### Spatial Attention block
Similar to the channel attention block, the spatial attention block runs average and max pooling across the channel dimension and concatenates them to create a spatial feature desciptor. A convolution layer is run over this to generate a spatial attention map Ms (F) ∈ RH×W
### ResBlock + CBAM
This paper also introduces a ResBlock+CBAM module that has a shortcut that adds the original input feature map back to enhanced output of CBAM followed by an activation function.

## ECA
The SENet channel attention module squeezes the spatial dimension using average pooling and runs a bottleneck of two fully connected layers. The bottleneck causes a dimensionality reduction around the non-linearity and forms the channel attention. This causes an indirect relation between the channel and its weights. ECA-Net improves upon this switching the the bottleneck a 1D convoultion layer in the channel dimension. This avoids the dimensionality reduction and still manages to capture local cross-channel interactions well. 

## HPECA-YOLOv8

## YOLOv8-AM

# Methodogy
## ICBAM YOLOv8 and YOLOv10

# Results
# Conclusion