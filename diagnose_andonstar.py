import cv2
import time

print("=" * 70)
print("ANDONSTAR CAMERA DIAGNOSTIC")
print("=" * 70)

for index in range(4):
    print()
    print(f"========== CAMERA INDEX {index} ==========")

    for backend_name, backend in [
        ("DirectShow", cv2.CAP_DSHOW),
        ("MSMF", cv2.CAP_MSMF),
    ]:
        print(f"\n--- Backend: {backend_name} ---")

        cap = cv2.VideoCapture(index, backend)

        print("Opened:", cap.isOpened())

        if not cap.isOpened():
            cap.release()
            continue

        # Start with a conservative resolution.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        time.sleep(1)

        print(
            "Reported:",
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "x",
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "FPS:",
            cap.get(cv2.CAP_PROP_FPS),
        )

        received = False

        for attempt in range(10):
            ret, frame = cap.read()

            print(
                f"Attempt {attempt + 1}: "
                f"ret={ret}, "
                f"shape={None if frame is None else frame.shape}"
            )

            if ret and frame is not None:
                received = True
                break

            time.sleep(0.2)

        if received:
            print(">>> SUCCESS: Frames received.")
        else:
            print(">>> FAILED: Camera opened but no frame received.")

        cap.release()

print()
print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)