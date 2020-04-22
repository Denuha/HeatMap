import cv2
import pandas
import numpy as np
import copy
import imutils
import sys

from imutils.video import VideoStream
from time import sleep

from datetime import datetime

#  https://github.com/bwsw/rt-motion-detection-opencv-python
#  https://www.geeksforgeeks.org/webcam-motion-detector-python/

args = sys.argv

video = cv2.VideoCapture('VID_20200326_170832.mp4') # filename
fps = video.get(cv2.CAP_PROP_FPS)
print("FPS video file: ", fps)

start_time = datetime.now()

static_back = None
motion_list = [None, None]
time = []
df = pandas.DataFrame(columns=["Start", "End"])

first_frame = None
ff = None

num_frame = 0
count_frames = 0
motion = 0
op = 0

history = []

while True:
    flag = False

    check, frame = video.read()

    if frame is None:
        break

    scale_percent = 50  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (31, 31), 0)
    if num_frame > fps/2:
        if motion == 1:
            history += (thresh_frame/3000)
            pass
        num_frame = 0

    if first_frame is None:
        first_frame = copy.deepcopy(gray)
        history = np.zeros(first_frame.shape)
        ff = copy.deepcopy(frame)
        continue

    diff_frame = cv2.absdiff(first_frame, gray)
    thresh_frame = cv2.threshold(diff_frame, 25, 255, cv2.THRESH_BINARY)[1]

    k = np.ones((2, 2), np.uint8)
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=4)

    cnts = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    frame1 = copy.deepcopy(frame)

    if num_frame == 0 and motion == 1:
        flag = True

    motion = 0
    for contour in cnts:
        count = cv2.contourArea(contour)
        if count < 5000/scale_percent:
            continue
        motion = 1
        (x, y, w, h) = cv2.boundingRect(contour)
        # making green rectangle around the moving object
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if flag:
        motion = 1

    cv2.imshow("history", history)

    if motion == 0:
        text = "Unoccupied"
    else:
        text = "Occupied"

    motion_list.append(motion)
    motion_list = motion_list[-2:]

    # Appending Start time of motion
    if motion_list[-1] == 1 and motion_list[-2] == 0:
        time.append(datetime.now())

    # Appending End time of motion
    if motion_list[-1] == 0 and motion_list[-2] == 1:
        time.append(datetime.now())

    cv2.putText(frame1, "Status: {}".format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Result', frame1)
    key = cv2.waitKey(1)
    if key == ord('q'):
        if motion == 1:
            time.append(datetime.now())
        break

    num_frame += 1
    count_frames += 1

# Appending time of motion in DataFrame
for i in range(0, len(time)-1):
    df = df.append({"Start": time[i], "End": time[i + 1]}, ignore_index=True)

history_heat = []
for i in range(history.shape[0]):
    row = []
    for j in range(history.shape[1]):
        a = history[i][j]*255
        if a > 255:
            b = min(a / 255 + 100, 255)
            row.append([0, a-b, b])
        else:
            row.append([0, history[i][j]*255, 0])
    history_heat.append(row)

history_hot = np.array(history_heat, 'uint8')
dist = cv2.add(ff, history_hot)
cv2.imshow("Heatmap", dist)

print(df)
end_time = datetime.now()
elapsed_time = end_time-start_time
print("Count frames:", count_frames)
print("Elapsed time:", elapsed_time)
print("Average frames processed per second:", count_frames/elapsed_time.seconds)

video.release()

cv2.waitKey(0)
cv2.destroyAllWindows()
