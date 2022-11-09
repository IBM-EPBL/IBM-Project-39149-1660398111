import mediapipe as mp
import numpy as np
from flask import Flask, render_template, request
import cv2
import os
from keras.models import load_model
from werkzeug.utils import secure_filename


app = Flask(__name__)
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/process", methods=['GET', 'POST'])
def process():
    if request.method == 'POST':
        upload_image = request.files['upload_image']
        basepath=os.path.dirname(__file__)
        file_path=os.path.join(basepath,'static',secure_filename(upload_image.filename))
        upload_image.save(file_path)
        model1 = load_model('gesture.h5')
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        mpDraw = mp.solutions.drawing_utils
        cap = cv2.VideoCapture(0)
        while True:
            _, frame = cap.read()

            h, w, c = frame.shape

            frame = cv2.flip(frame, 1)
            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = hands.process(framergb)
            res = ''

            if result.multi_hand_landmarks:
                landmarks = []
                for handslms in result.multi_hand_landmarks:
                    x_max = 0
                    y_max = 0
                    x_min = w
                    y_min = h
                    for lm in handslms.landmark:
                        x = int(lm.x * w)
                        y = int(lm.y * h)

                        landmarks.append([x, y])
                        if x > x_max:
                            x_max = x
                        if x < x_min:
                            x_min = x
                        if y > y_max:
                            y_max = y
                        if y < y_min:
                            y_min = y
                    cv2.rectangle(frame, (x_min - 5, y_min - 5), (x_max + 5, y_max + 5), (0, 255, 0), 2)
                    framegray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    hand = framegray[y_min - 5:y_max + 5, x_min - 5:x_max + 5]
                    hand = cv2.resize(hand, (128, 128))
                    hand = hand / 255
                    hand = hand.reshape(128, 128, 1)
                    hand = np.expand_dims(hand, axis=0)
                    mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
                    prediction = model1.predict(hand)
                    res = np.argmax(prediction)
                    image1 = cv2.imread(file_path)
                    image1= cv2.resize(image1,(300,300))
                    if res==1:
                        resized = cv2.resize(image1, (200, 200))
                        cv2.imshow("Resizing", resized)
                        key=cv2.waitKey(3000)

                        if (key & 0xFF) == ord("1"):
                            cv2.destroyWindow("Resizing")

                    elif res==2:
                        blurred = cv2.GaussianBlur(image1, (21, 21), 0)
                        cv2.imshow("Blurred", blurred)
                        key=cv2.waitKey(3000)
                        if (key & 0xFF) == ord("3"):
                            cv2.destroyWindow("Blurred")

                    elif res==3:
                        (h, w, d) = image1.shape
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, -45, 1.0)
                        rotated = cv2.warpAffine(image1, M, (w, h))
                        cv2.imshow("OpenCV Rotation", rotated)
                        key=cv2.waitKey(3000)
                        if (key & 0xFF) == ord("2"):
                            cv2.destroyWindow("OpenCV Rotation")

                    
                    elif res==4:
                        cv2.rectangle(image1, (480, 170), (650, 420), (0, 0, 255), 2)
                        cv2.imshow("Rectangle", image1)
                        cv2.waitKey(0)
                        key=cv2.waitKey(3000)
                        if (key & 0xFF) == ord("0"):
                            cv2.destroyWindow("Rectangle")


                    else:
                        continue

            
            cv2.imshow("Output", frame)

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    return render_template("index.html")


@app.route("/intro")
def intro_page():
    return render_template("intro.html")

@app.route("/index")
def index_page():
    return render_template("index.html")
