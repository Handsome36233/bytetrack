import cv2
import numpy as np
import onnxruntime as ort
from detector import *
from bytetrack import BYTETracker
import argparse


parser = argparse.ArgumentParser(description="ONNX 模型和视频文件路径输入")
parser.add_argument('--onnx_model', type=str, required=True, help="ONNX 模型文件的路径")
parser.add_argument('--video_path', type=str, required=True, help="视频文件的路径")
args = parser.parse_args()
onnx_model = args.onnx_model
video_path = args.video_path

low_conf_thresh = 0.2
high_conf_thresh = 0.6
iou_thres = 0.45

tracker = BYTETracker(frame_rate=30, low_conf_thresh=low_conf_thresh, high_conf_thresh=high_conf_thresh)

session = ort.InferenceSession(onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
model_inputs = session.get_inputs()
input_shape = model_inputs[0].shape
input_height = input_shape[2]
input_width = input_shape[3]

cap = cv2.VideoCapture(video_path)
while True:
    ret, img = cap.read()
    if not ret:
        break
    original_img = img.copy()
    img_height, img_width = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_width, input_height))
    image_data = np.array(img) / 255.0
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    outputs = session.run(None, {model_inputs[0].name: image_data})
    bboxs = postprocess(outputs, img_width, img_height, input_width, input_height, iou_thres, low_conf_thresh)

    output_stracks = tracker.update(bboxs)
    for track in output_stracks:
        tlwh = list(map(int, track.tlwh()))
        cv2.rectangle(original_img, (tlwh[0], tlwh[1]), (tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]), (0, 255, 0), 2)
        cv2.putText(original_img, str(track.track_id), (tlwh[0], tlwh[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("result", original_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

