import cv2 as cv
import numpy as np

cap = cv.VideoCapture('Ball_Tracking.mp4')

positions = []   # stores ball centers
canvas = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if canvas is None:
        canvas = np.zeros_like(frame)

    blur = cv.GaussianBlur(frame, (9,9), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

    lower = np.array([40, 50, 50])
    upper = np.array([80, 255, 255])

    mask = cv.inRange(hsv, lower, upper)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        ball = max(contours, key=cv.contourArea)
        (x, y), r = cv.minEnclosingCircle(ball)

        if r > 10:
            center = (int(x), int(y))
            positions.append(center)

            cv.circle(frame, center, int(r), (0,255,0), 2)

            if len(positions) > 1:
                cv.line(canvas, positions[-2], positions[-1], (255,0,0), 3)

    result = cv.addWeighted(frame, 1, canvas, 1, 0)

    cv.imshow("Ball Path", result)
    cv.imshow("Mask", mask)

    if cv.waitKey(20) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

