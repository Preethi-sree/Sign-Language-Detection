import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3
import tkinter as tk
from tkinter import Button
from PIL import Image, ImageTk

# Initialize video capture, hand detector, and classifier
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
labels = ["A", "B", "C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","HELLO","I LOVE YOU","I REALLY LOVE YOU","YES","NO","THANKS","DONE","SEE YOU LATER"]

# Initialize text-to-speech engine
text_speech = pyttsx3.init()

# Initialize the last prediction to None

def close_window():
    global root
    root.quit()
    cap.release()
    cv2.destroyAllWindows()
last_prediction = None

def speak_prediction():
    global last_prediction
    if last_prediction is not None:
        text_speech.say(last_prediction)
        text_speech.runAndWait()

# Create a Tkinter window
root = tk.Tk()
root.title("Hand Gesture Recognition")

# Create a label to show the video
label = tk.Label(root)
label.pack()

# Create a frame for buttons
button_frame = tk.Frame(root)
button_frame.pack(side=tk.BOTTOM, padx=10, pady=10)

# Create a button to close the window
quit_button = Button(button_frame, text="Quit", command=close_window)
quit_button.pack(side=tk.LEFT, padx=5)

# Create a button to convert output to speech
speak_button = Button(button_frame, text="Speak", command=speak_prediction)
speak_button.pack(side=tk.LEFT, padx=5)

def show_frame():
    global last_prediction
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        if imgCrop.shape[0] == 0 or imgCrop.shape[1] == 0:
            return  # Skip empty crop

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        try:
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            last_prediction = labels[index]  # Store the last prediction
            print(prediction, index)
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

    # Convert the image to a format Tkinter can display
    imgRGB = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
    imgPIL = Image.fromarray(imgRGB)
    imgTk = ImageTk.PhotoImage(image=imgPIL)

    # Update the label with the new image
    label.imgTk = imgTk
    label.configure(image=imgTk)

    # Call this function again after 10 milliseconds
    root.after(10, show_frame)

# Start the video loop
show_frame()

# Run the Tkinter main loop
root.mainloop()
