import cv2

from ultralytics.cfg import YAML
from yolo_segmentation import YOLO_Segmentation
from yolo_segmentation import YOLO_Detection
import time

detected_fruits = set()
fruit_counts = {"apple": 0, "banana": 0, "orange": 0}
total_price = 0
fruit_count = 0
fruit_price = {"apple": 1, "banana": 2, "orange": 3}

yaml_loader = YAML()
data = yaml_loader.load('coco128.yaml')
class_list = data['names']

ys = YOLO_Segmentation("yolov8m-seg.pt")

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

font = cv2.FONT_HERSHEY_PLAIN

while True:
    img = cv2.imread('Resources/fruits2.0.jpg')
    startTime = time.time()

    # reset counter
    fruit_counts = {"apple": 0, "banana": 0, "orange": 0}
    total_price = 0
    fruit_count = 0

    bboxes, classes, segmentations, scores = ys.detect(img)

    for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
        (x, y, x2, y2) = bbox
        if score > 0.8 and class_id in [46, 47, 49]:  # apple, banana, orange
            fruit_name = class_list[class_id]
            confidence_percent = int(score * 100)

            cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f'{fruit_name} {confidence_percent}%',
                        (x, y - 10), font, 1, (0, 0, 255), 2)

            # count
            fruit_counts[fruit_name] += 1

    # calculate totals
    for fruit, count in fruit_counts.items():
        fruit_count += count
        total_price += fruit_price.get(fruit, 0) * count

    # Show fruit totals in top-left corner
    cv2.putText(img, f'Total Fruits: {fruit_count}', (20, 10),
                font, 1, (255, 0, 100), 2)
    cv2.putText(img, f'Total Price: ${total_price}', (20, 40), font,
                1, (255, 0, 100), 2)

    # Show per-fruit breakdown with subtotal prices
    breakdown_text = f"Apples: {fruit_counts['apple']} (${fruit_counts['apple'] * fruit_price['apple']}), " \
                     f"Oranges: {fruit_counts['orange']} (${fruit_counts['orange'] * fruit_price['orange']}), " \
                     f"Bananas: {fruit_counts['banana']} (${fruit_counts['banana'] * fruit_price['banana']})"
    cv2.putText(img, breakdown_text, (20, 60), font,
                1, (255, 100, 255), 2)

    # Show FPS
    newTime = time.time()
    FPS = str(int(1 / (newTime - startTime)))
    cv2.putText(img, f'FPS: {FPS}', (img.shape[1] - 150, 30), font,
                1, (255, 0, 0), 2)

    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()