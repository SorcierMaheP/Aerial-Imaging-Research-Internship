# Onboard Classification of Tree Species using RGB UAV Imagery

## Datasets
The new and improved Bi-Class Dataset we created, which features
two classes, namely; Banana and Coconut, with 11.6k images
of Banana trees, and 14.4k images of Coconut trees.  
BiClass Dataset: https://drive.google.com/file/d/1YzVk-7L4Dj0rf303fNbhq1Hqw14LpEEk/view?usp=drive_link 

<<<<<<< HEAD
# To Use the New Models
Before using the new models, the new modules have to be added to your local ultralytics package. To do this, simply run `editCode.py`.
This will add the HyP-ECA module to the package, along with the yaml file for YOLOv8 HyP-ECA.
To add the ICBAM module, run `editCodeICBAM.py` AFTER running `editCode.py`.
To add the ResBlock CBAM module, run `edit_code_ResBlock_CBAM.py` AFTER running `editCode.py`.
To add the yaml file for YOLOv10 ICBAM, run `addICBAMv10.py` after runnning `editCodeICBAM.py`
=======
## To Use the New Models
Before using the new models, the new modules have to be added to your local ultralytics package. To do this, simply run '''editCode.py'''.  
This will add the HyP-ECA module to the package, along with the yaml file for YOLOv8 HyP-ECA.  
To add the ICBAM module, run '''editCodeICBAM.py''' AFTER running '''editCode.py'''.  
To add the ResBlock CBAM module, run '''edit_code_ResBlock_CBAM.py''' AFTER running '''editCode.py'''.  
To add the yaml file for YOLOv10 ICBAM, run '''addICBAMv10.py''' after runnning '''editCodeICBAM.py'''  
  
#### NOTE: Run these codes only ONCE. Running them multiple times will add the modules again, which could lead to issues.  
>>>>>>> 2fb6903606a715401bce0e8119b093867bc0b390

## To Train the Models
Run the following command in terminal:
    `yolo detect train data=<path/to/data.yaml> model=<name_of_pretrained_model.pt> epochs=100 imgsz=640 batch=0.85 patience=10 plots=True`


# TODO
- [ ] Add a requirements.txt
- [ ] Add benchmark values, maybe?

# DONE
- [x] Fix the dataset anomalies causing errors in the faster-RNN finetune code
- [x] Benchmark many models on a raspberry pi 4 4GB version
- [-] Use two stage object detection models with HARR cascades 
