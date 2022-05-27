
from asyncio.subprocess import PIPE
from pyexpat import model
from random import randint
from sys import stdout
import sys
import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time

from tkinter import *
from PIL import ImageTk, Image
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import json
import cv2
import numpy as np
from utils import face_extractor, update_attendance
from Generate_Dataset import train
import tkinter as tk
from tkinter import ttk
import subprocess


personName = NONE

name = ""


class tkinterApp(tk.Tk):
     
    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)
        self.geometry("800x640")
        self.container = ttk.Frame(self)
        
        photo = PhotoImage(file = "assets/icon.ico")
        self.iconphoto(False, photo)

        self.title('Trabia - Face based Entry Logger')

        style = ttk.Style(self)
        self.tk.call('source', 'azure.tcl')
        self.tk.call("set_theme", "azure")

        self.container.pack(side = "top", fill = "both", expand = True)
  
        self.container.grid_rowconfigure(0, weight = 1)
        self.container.grid_columnconfigure(0, weight = 1)
  
        self.frames = {} 

        self.show_frame(AttendancePage)
  
    def show_frame(self, cont):
                
        frame=cont(self.container, self)
        self.frames[cont] = frame
        for frame in self.frames:
            if(frame == cont):
                self.frames[frame].pack()
            else:
                self.frames[frame].destroy()
  


class AttendancePage(tk.Frame):
    def pack_forget(self):
        self.canvas.pack_forget()
        self.btn_snapshot.pack_forget()
        ttk.Frame.pack_forget(self)

    def destroy(self):
        self.canvas.destroy()
        self.btn_snapshot.destroy()
        self.btn1.destroy()
        self.btn2.destroy()
        ttk.Frame.destroy(self)

    def __init__(self, window, controller, video_source=0):
        super().__init__(window)
        
        self.window = window
        global name
        self.video_source = video_source
        self.vid = MyVideoCapture(self.video_source)

        self.canvas = Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()
         
        self.btn_snapshot = ttk.Button(window, text="Register Entry", command= lambda: update_attendance(name))
        
        params = dict(fill=X, padx=200, pady=10)
        self.btn_snapshot.pack(**params,  ipady=20)
        self.btn1 = ttk.Button(window, text ="Add Face", command = lambda : controller.show_frame(AddFace))
        self.btn1.pack(**params)
        self.btn2 = ttk.Button(window, text ="Exit App", command = lambda : sys.exit(0))
        self.btn2.pack(**params)

        self.delay = 15
        self.update_widget()

    
    def update_widget(self):

        ret, frame = self.vid.get_frame()
        
        if ret:
            self.image = PIL.Image.fromarray(frame)
            self.photo = PIL.ImageTk.PhotoImage(image=self.image)
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
        
        self.window.after(self.delay, self.update_widget)



class AddFace(tk.Frame):
     
    def __init__(self, parent, controller):
         
        ttk.Frame.__init__(self, parent)
        
        global personName
        params = dict(fill=X, padx=200, pady=5)
        self.statusLabel = ttk.Label(self, text = "Press Start to begin adding faces", font=("Arial", 14))
        ttk.Label(self, text ="Add faces to recognition database", font=("Arial", 24)).pack()
        img = ImageTk.PhotoImage(Image.open("assets/crowd.jpg").resize((600,400)))
        il = Label(self, image=img)
        il.photo = img
        il.pack()
        personName = ttk.Entry(self, font=("Arial", 24))
        personName.insert(0,"Enter your name")
        personName.pack(pady= 15)
        self.statusLabel.pack()
        ttk.Button(self, text="Capture Face", command= self.trainStart).pack(**params, ipady=5)
        ttk.Button(self, text = "Take to training page.", command = lambda : controller.show_frame(TrainRecognitionModel)).pack(**params)


    def trainStart(self):
        self.statusLabel.config(text = "Processing")
        global personName
        text = personName.get()
        train(text)
        self.statusLabel.config(text = "Done")

class TrainRecognitionModel(tk.Frame):
    
    def destroy(self):
        tk.Frame.destroy(self)
        self.il.destroy()
        self.statusLabel.destroy()
        self.labelTitle.destroy()
        self.b1.destroy()
        self.b2.destroy()
        self.b3.destroy()

    def __init__(self, parent, controller):
         
        ttk.Frame.__init__(self, parent)
        self.labelTitle = ttk.Label(parent, text ="Train to recognize", font=("Arial", 24))
        self.labelTitle.pack()
        
        img = ImageTk.PhotoImage(Image.open("assets/network.png").resize((600,400)))
        self.il = Label(parent, image=img)
        il = self.il
        il.photo = img
        il.pack()
        
        self.statusLabel = ttk.Label(parent, text = "Press Start to train the model", font=("Arial", 14))
        
        self.statusLabel.pack()
        params = dict(fill=X, padx=200, pady=5)

        self.b1 = ttk.Button(parent, text="Start Training", command = self.train_model)
        self.b1.pack(**params, ipady=5)
        self.b2 = ttk.Button(parent, text = "Add Another Face", command = lambda : controller.show_frame(AddFace))
        self.b2.pack(**params)
        self.b3 = ttk.Button(parent, text = "Home", command = lambda : controller.show_frame(AttendancePage))
        self.b3.pack(**params)


    def train_model(self):
        self.statusLabel.config(text="Training...")
        try:
            proc = subprocess.Popen("powershell.exe python train.py", creationflags=subprocess.CREATE_NEW_CONSOLE)
            
            proc.wait()
            proc.communicate()
            if(proc.returncode != 0): 
                raise ValueError("Command Exit with error")
        except:
            self.statusLabel.config(text="Error")
        else:
            self.statusLabel.config(text="Training Complete")
  


class App:
    def __init__(self, window, window_title, video_source1=0):
        self.window = window
        self.window.title(window_title)

        self.vid1 = AttendancePage(window, video_source1)
        self.vid1.pack()
        
        self.window.mainloop()
     

class MyVideoCapture:
    def __init__(self, video_source=0):
        
        classF = open('classes.json')
        self.classes = json.load(classF)
        classes = self.classes

        print("Available Classes", classes)
        self.model = load_model('facefeatures_new_model.h5')
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
    
        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
        self.width = 600
        self.height = 450
    
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            classes = self.classes
            model = self.model
            
            face=face_extractor(frame)
            global name
            name="None matching"

            if type(face) is np.ndarray:
                face = cv2.resize(face, (224, 224))
                im = Image.fromarray(face, 'RGB')
                img_array = np.array(im)
                img_array = np.expand_dims(img_array, axis=0)
                pred = model.predict(img_array)
                predClass = pred[0].argmax()
                classFinal = str(predClass)
                print("Class: ", classFinal)
                print("Pred: ", pred[0])
            
            
                if(classFinal in classes and pred[0][predClass] > 0.5):
                    name = classes[classFinal]
                    
                cv2.putText(frame,name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            else:
                cv2.putText(frame,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

            if ret:
                frame= cv2.resize(frame, (600,450))

                # frame = cv2.resize(frame, (400, 300))
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)
    
    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
 
# Create a window and pass it to the Application object
# App(tkinter.Tk(), "Tkinter and OpenCV", 0)
app = tkinterApp()
app.mainloop()