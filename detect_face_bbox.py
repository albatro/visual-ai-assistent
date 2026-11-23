"""Find face bbox in a single frame using MediaPipe — to plan a video crop."""
import cv2
import mediapipe as mp

img = cv2.imread("_peter_frame.jpg")
h, w = img.shape[:2]
print(f"Frame size: {w}x{h}")

# Try BlazeFace short-range first (closer), then full-range
for model in (0, 1):
    fd = mp.solutions.face_detection.FaceDetection(
        model_selection=model, min_detection_confidence=0.3
    )
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = fd.process(rgb)
    if res.detections:
        for d in res.detections:
            box = d.location_data.relative_bounding_box
            x = int(box.xmin * w)
            y = int(box.ymin * h)
            bw = int(box.width * w)
            bh = int(box.height * h)
            print(f"model={model} score={d.score[0]:.2f}  bbox=({x},{y},{bw}x{bh})")
        break
    else:
        print(f"model={model}: no faces")
