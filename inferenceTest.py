from ultralytics import YOLO
import cv2
import math
from time import time
import torch

# model
model = YOLO("Models/yolov8n_improvedECA_best.pt")
model.export(format="ncnn")
model = YOLO("Models/yolov8n_improvedECA_best_ncnn_model")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')

# object classes
classNames = ["coconut", "coniferous", "date palm", "decidious", "banana"]

totalTime = 0
frameCount = 0

videoNames = ['one_c.mp4', "two_c.mp4",
              "three_c.mp4", "four_c.mp4", "five_c.mp4"]

for name in videoNames:

    cap = cv2.VideoCapture(f"Videos/{name}")
    cap.set(3, 640)
    cap.set(4, 480)

    start = time()
    frame = 0

    while True:

        print(frame)
        frame += 1
        success, img = cap.read()

        if not success:
            break
        results = model(img, stream=True, device=device)

        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(
                    x2), int(y2)  # convert to int values

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                # print("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0])
                # print("Class name -->", classNames[cls])

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

    frameCount += frame

    end = time()
    # print("Total time = ", (end - start))
    totalTime += (end-start)

    cap.release()

avgTime = totalTime / 5
print("\n\nTotal: ", totalTime, "s")
print("Total Frames: ", frameCount)
timePerFrame = ((avgTime / frameCount) * 1000)

print("Avg per frame: ", timePerFrame, "ms")
print("FPS: ", (1 / timePerFrame) * 1000, "fps")
