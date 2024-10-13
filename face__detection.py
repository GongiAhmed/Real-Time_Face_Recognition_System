import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
import time
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import PhotoImage
from PIL import Image, ImageTk


def collect_data():
    # Exécutez le script de collecte de données (collectdata.py) en utilisant os.system
    os.system("python datacollect.py")


def train_model():
    # Exécutez le script d'entraînement du modèle (trainingdemo.py) en utilisant os.system
    os.system("python trainingdemo.py")

# Fonction pour afficher la vidéo en direct dans la fenêtre tkinter
def show_video():
    _, frame = cap.read()
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
    for x,y,w,h in faces:
        img = rgb_img[y:y+h, x:x+w]
        img = cv.resize(img, (160,160)) # 1x160x160x3
        img = np.expand_dims(img, axis=0)
        ypred = facenet.embeddings(img)
        face_name = model.predict(ypred)
        final_name = encoder.inverse_transform(face_name)[0]
        cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 10)
        cv.putText(frame, str(final_name), (x,y-10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0,0,255), 3, cv.LINE_AA)
        

    photo = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
    photo = Image.fromarray(photo)
    photo = ImageTk.PhotoImage(image=photo)
    label.config(image=photo)
    label.image = photo
    label.after(10, show_video)

# Fonction pour quitter l'application
def quit_app():
    if messagebox.askyesno("Quitter", "Êtes-vous sûr de vouloir quitter ?"):
        cap.release()
        cv.destroyAllWindows()
        root.destroy()

# Créez la fenêtre principale tkinter
root = tk.Tk()
root.title("Reconnaissance faciale")

# Créez un canevas pour afficher la vidéo en direct
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

# Créez une étiquette pour afficher la vidéo
label = tk.Label(canvas)
label.pack()

# Créez un bouton pour quitter l'application
quit_button = ttk.Button(root, text="Quitter", command=quit_app)
quit_button.pack()

# Bouton pour collecter les données
collect_button = ttk.Button(root, text="Collect Data", command=collect_data)
collect_button.pack()

# Bouton pour entraîner le modèle
train_button = ttk.Button(root, text="Train Model", command=train_model)
train_button.pack()

# Initialisez la capture vidéo
cap = cv.VideoCapture(0)

# Initialisation des variables pour la capture d'inconnu
count = 0
unknown_capture_interval = 3000
last_unknown_capture_time = 0

# Initialisez les modèles pour la reconnaissance faciale
facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
model = pickle.load(open("svm_model.pkl", 'rb'))

# Afficher la vidéo en direct dans la fenêtre tkinter
show_video()

# Exécutez la boucle principale tkinter
root.mainloop()
