from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class BBox:
    left: int
    top: int
    width: int
    height: int
    class_id: int
    score: float

def postprocess(outputs, img_width, img_height, input_width, input_height, iou_thres, confidence_thres):
    outputs = np.transpose(np.squeeze(outputs[0]))
    rows = outputs.shape[0]
    boxes = []
    scores = []
    class_ids = []
    x_factor = img_width / input_width
    y_factor = img_height / input_height
    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)
        if max_score >= confidence_thres:
            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])
    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)
    return [BBox(*boxes[i], class_ids[i], scores[i]) for i in indices]
