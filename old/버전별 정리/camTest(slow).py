from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2

#캠 키기
stream = cv2.VideoCapture(0)
fps = FPS().start()

while True:
    grabbed, frame = stream.read()
    if not grabbed:
        break

    frame = imutils.resize(frame, width=450)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.dstack([frame, frame, frame])

    cv2.putText(frame, "Slow Method", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # show the frame and update the FPS counter
    cv2.imshow("Frame", frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    fps.update()

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
fps.stop()