#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:07:17 2020

@author: Shane Zhong
@reference: https://github.com/jy6zheng/FacialExpressionRecognition
"""
from scipy.spatial import distance as dist
import cv2
from imutils import face_utils
from imutils.video import VideoStream
from fastai.vision import *
import imutils
import argparse
import time
import dlib
from datetime import datetime
import matplotlib.pyplot as plt
import os
from pygame import mixer
from src.functions import plot_emotion_charts, trigger_sound

# Capture start time
start_datetime = datetime.now()
start_time = start_datetime.strftime("%Y-%m-%d_%H:%M:%S")
print(start_datetime)
print(start_time)

# User inputs
ap = argparse.ArgumentParser()
ap.add_argument("--save-video", dest="save-video", action="store_true", default=False)
ap.add_argument("--save-csv", dest="save-csv", action="store_true", default=False)
ap.add_argument("--camera_id",
                help="The webcam id, for MacBook it is default to 1",
                dest="camera_id",
                default='1')
ap.add_argument("--low-fps", dest="low-fps", action="store_true", default=False)
args = vars(ap.parse_args())

# Set the code directory
print(os.getcwd())

# Set path to your project directory
path = '/Users/szhong/Documents/GitHub/EmotionMonitor/'

# Set up directories
model_dir = 'src/model/'
classifier_dir = path + model_dir + "haarcascade_frontalface_default.xml"
learner_dir = path + model_dir
shape_predictor_dir = path + model_dir + "shape_predictor_68_face_landmarks.dat"

sound_dir = path + 'src/sound/sound1.mp3'
SOUND_TIME_INTERVAL = 20  # in seconds

learn = load_learner(path=learner_dir, file='export.pkl')
face_cascade = cv2.CascadeClassifier(classifier_dir)

# Turn on camera and start capturing
vs = VideoStream(src=int(args["camera_id"]), framerate=1).start()  # MacBook scr=1
start = time.perf_counter()

data = []
time_value = 0

EYE_AR_THRESH = 0.20
EYE_AR_CONSEC_FRAMES = 10

COUNTER = 0


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def data_time(time_value, prediction, probability, ear):
    current_time = int(time.perf_counter()-start)
    if current_time != time_value:
        data.append([current_time, prediction, probability, ear])
        time_value = current_time
    return time_value


# Load model input
predictor = dlib.shape_predictor(shape_predictor_dir)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


if args["save-video"]:
    out = cv2.VideoWriter(path + "output/"+start_time+"liveoutput.avi",
                          cv2.VideoWriter_fourcc('M','J','P','G'), 10, (450,253))

# Initalise charts
plt.rcParams['figure.figsize'] = [10, 8]
fig, axs = plt.subplots(5)
if args['low-fps']:
    chart_refresh_rate = 10
else:
    chart_refresh_rate = 2

# Modelling loop
while True:
    if args['low-fps']:
        time.sleep(0.6)

    frame = vs.read()
    frame = imutils.resize(frame, width=450)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coord = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    for coords in face_coord:
        X, Y, w, h = coords
        H, W, _ = frame.shape
        X_1, X_2 = (max(0, X - int(w * 0.3)), min(X + int(1.3 * w), W))
        Y_1, Y_2 = (max(0, Y - int(0.3 * h)), min(Y + int(1.3 * h), H))
        img_cp = gray[Y_1:Y_2, X_1:X_2].copy()
        prediction, idx, probability = learn.predict(Image(pil2tensor(img_cp, np.float32).div_(225)))
        # print(prediction, idx, probability)

        cv2.rectangle(
                img=frame,
                pt1=(X_1, Y_1),
                pt2=(X_2, Y_2),
                color=(128, 128, 0),
                thickness=2,
            )
        rect = dlib.rectangle(X, Y, X+w, Y+h)

        cv2.putText(frame, str(prediction), (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 255, 255), 2)

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the (x, y)-coordinates for the facial landmarks
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        # Draw the shape of eyes
        # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "Distracted", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
        cv2.putText(frame, "Eye Ratio: {:.2f}".format(ear), (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        time_value = data_time(time_value, prediction, probability.tolist(), ear)

        # Plot emotion charts
        if (len(data) != 0) and (int(time_value) % chart_refresh_rate == 0):
            plot_emotion_charts(data, start_datetime, axs)

        # Check anxiety level and trigger sound if avg sad/angry over the past 20 seconds are greater than 0.5
        if (len(data) != 0) and (int(time_value) % SOUND_TIME_INTERVAL == 0):
            trigger_sound(sound_dir, SOUND_TIME_INTERVAL, data)


    cv2.imshow("frame", frame)

    if args["save-video"]:
        out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

if args["save-csv"]:
    df = pd.DataFrame(data, columns = ['Time (seconds)', 'Expression', 'Probability', 'EAR'])
    df.to_csv(path+'output/'+start_time+'_exportlive.csv', index=False)
    print("model saved to exportlive.csv")

vs.stop()
if args["save-video"]:
    print("done saving video")
    out.release()
cv2.destroyAllWindows()
