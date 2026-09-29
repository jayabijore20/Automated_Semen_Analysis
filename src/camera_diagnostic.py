import cv2
import time

print("===== TEST CAMERA 0 =====")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Opened:", cap.isOpened())

time.sleep(2)

ret, frame = cap.read()

print("ret =", ret)

if frame is not None:
    print("Shape =", frame.shape)

if ret:
    while True:
        cv2.imshow("Camera 0", frame)

        if cv2.waitKey(1) == 27:
            break

        ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()

print("Finished.")