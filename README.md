# Onboard Classification of Tree Species using RGB UAV Imagery

## Performance
Working of YOLOv8s ICBAM on a sample video with dense coconut plantations.

https://github.com/user-attachments/assets/ec52e710-1269-45cb-aa83-eed2f2e723c7

## Datasets
### MTAD
The original Multi-Tree Species Aerial Detection (MTAD) Dataset created by Anish Natekar et al., consisting of five classes, namely; Banana, Coconut, Date Palm, Deciduous and Coniferous. This was very imbalanced, and except banana, all classes had around 1.5k instances, while the recommended minimum amount is 10k <a href="https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/">instances per class</a>.
### Bi-Class
The new and improved Bi-Class Dataset we created, which features
two classes, namely; Banana and Coconut, with 11.6k images
of Banana trees, and 14.4k images of Coconut trees.  
BiClass Dataset: https://drive.google.com/file/d/1YzVk-7L4Dj0rf303fNbhq1Hqw14LpEEk/view?usp=drive_link 

## To Use the New Models
- Before using the new models, the new modules have to be added to your local ultralytics package. To do this, simply run `editCode.py`.  This will add the HyP-ECA module to the package, along with the yaml file for YOLOv8 HyP-ECA.  
- To add the ICBAM module, run `editCodeICBAM.py` **AFTER** running `editCode.py`.  
- To add the ResBlock CBAM module, run `edit_code_ResBlock_CBAM.py` **AFTER** running `editCode.py`.  
- To add the yaml file for YOLOv10 ICBAM, run `addICBAMv10.py` **AFTER** runnning `editCodeICBAM.py`  
  
#### NOTE: Run these codes only ONCE. Running them multiple times will add the modules again, which could lead to issues.  

## To Train the Models
Run the following command in terminal:  
```
yolo detect train \
data=<path/to/data.yaml> \
model=<name_of_pretrained_model.pt> \
epochs=100 imgsz=640 batch=0.85 patience=10 plots=True
```
