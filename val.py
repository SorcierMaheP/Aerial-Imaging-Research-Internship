from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    # model = YOLO("yolov8n.pt")  # load an official model
    model = YOLO("Models/yolov8n_improvedECA_best_ncnn_model")  # load a custom model

    # Validate the model
    metrics = model.val(data="C:/Users/PC/Downloads/augmented_dataset/Trspeclass.v2i.yolov8/data.yaml",
                        split="test")  # no arguments needed, dataset and settings remembered
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list contains map50-95 of each category
