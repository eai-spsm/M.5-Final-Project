from pathlib import Path

import cv2 as cv
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent / "data" / "best.pt"
CONF_THRESHOLD = 0.5
IMG_SIZE = 320       # smaller inference size = faster on Pi CPU
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
SKIP_FACTOR = 2       # only run inference on every (SKIP_FACTOR + 1)th frame
BOX_COLOR = (0, 255, 0)


def main():
    try:
        model = YOLO(str(MODEL_PATH))
    except Exception as e:
        print(f"Model Load Error: {e}")
        return

    try:
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera device at index 0.")

        cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    except Exception as e:
        print(f"Camera Initialization Error: {e}")
        return

    print("Starting video feed. Press 'q' or Ctrl+C to quit.")

    frame_count = 0
    last_boxes = []      # cached (x1, y1, x2, y2, label) from the last inference frame
    last_detected = set()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to grab frame from camera.")
                break

            if frame_count % (SKIP_FACTOR + 1) == 0:
                try:
                    results = model(frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)
                    boxes = results[0].boxes

                    detected = set()
                    last_boxes = []
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        name = model.names[cls_id]
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        last_boxes.append((x1, y1, x2, y2, f"{name} {conf:.2f}"))
                        detected.add(name)
                        if name not in last_detected:
                            print(f"Detected {name} ({conf:.2f})")
                    last_detected = detected

                except Exception as frame_err:
                    print(f"Error processing frame: {frame_err}")

            for x1, y1, x2, y2, label in last_boxes:
                cv.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 1)
                cv.putText(frame, label, (x1, max(y1 - 5, 0)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.4, BOX_COLOR, 1)

            frame_count += 1
            cv.imshow('YOLO Detection', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                print("Exiting on user request...")
                break

    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"Unexpected Runtime Error: {e}")

    finally:
        print("Cleaning up resources...")
        try:
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            cv.destroyAllWindows()
        except Exception as e:
            print(f"Error releasing OpenCV resources: {e}")


if __name__ == "__main__":
    main()
