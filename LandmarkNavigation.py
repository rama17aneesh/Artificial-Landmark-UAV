from tkinter import *
import tkinter
import numpy as np
import imutils
import cv2
from pyzbar import pyzbar
import os
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import model_from_json
from keras.preprocessing import image

main = tkinter.Tk()
main.title("Artificial Landmark-based Indoor Navigation System for an Autonomous Unmanned Aerial Vehicle")
main.geometry("800x500")

global inception_model
global video

def loadModel():
    global inception_model
    if os.path.exists('model/inception_weights.h5'):
        with open('model/inception.json', "r") as json_file:
            loaded_model_json = json_file.read()
            inception_model = model_from_json(loaded_model_json)
        inception_model.load_weights("model/inception_weights.h5")    
        #inception_model = load_model("model/inception_weights1.h5")
        print(inception_model.summary())
        pathlabel.config(text="          Inception Model Generated Successfully")
    else:
        input = Input(shape=(224, 224, 3))
        base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=input, input_shape=(299, 299, 3), pooling='avg', classes=1000)
        for l in base_model.layers:
            l.trainable = False
        t = base_model(input)
        o = Dense(2, activation='softmax')(t)
        inception_model = Model(inputs=input, outputs=o)
        inception_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        train_datagen = ImageDataGenerator()
        test_datagen = ImageDataGenerator()
        training_set = train_datagen.flow_from_directory('dataset/train',target_size = (224, 224), batch_size = 2, class_mode = 'categorical', shuffle=True)
        test_set = test_datagen.flow_from_directory('dataset/test',target_size = (224, 224), batch_size = 2, class_mode = 'categorical', shuffle=False)
        inception_model.fit_generator(training_set,samples_per_epoch = 40,nb_epoch = 1,validation_data = test_set,nb_val_samples = 10)
        inception_model.save_weights('model/inception_weights.h5')
        model_json = inception_model.to_json()
        with open("model/inception.json", "w") as json_file:
            json_file.write(model_json)
        print(model.summary())    
        pathlabel.config(text="          Inception Model Generated Successfully")
    
    
    
def upload():
    global video
    filename = filedialog.askopenfilename(initialdir="Video")
    pathlabel.config(text="          Video loaded")
    video = cv2.VideoCapture(filename)
    

def detectBarcode(image):
    barcodes = pyzbar.decode(image)
    for barcode in barcodes:
        (x, y, w, h) = barcode.rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
        text = "{} ({})".format(barcodeData, barcodeType)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
    return image

def landmarkDetection():
        while(True):
            ret, frame = video.read()
            if ret == True:
                cv2.imwrite("test.jpg",frame)
                imagetest = image.load_img("test.jpg", target_size = (224,224))
                imagetest = image.img_to_array(imagetest)
                imagetest = np.expand_dims(imagetest, axis = 0)
                predict = inception_model.predict(imagetest)
                frame = cv2.resize(frame,(500,500))
                frame = detectBarcode(frame)
                cv2.imshow("video frame", frame)
                if cv2.waitKey(800) & 0xFF == ord('q'):
                   break
            else:
                break
        video.release()
        cv2.destroyAllWindows()


def exit():
    global main
    main.destroy()
  

font = ('times', 14, 'bold')
title = Label(main, text='Artificial Landmark-based Indoor Navigation System for an Autonomous Unmanned Aerial Vehicle',anchor=W, justify=LEFT)
title.config(bg='black', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 14, 'bold')
loadButton = Button(main, text="Generate & Load Landmark Detection Model", command=loadModel)
loadButton.place(x=50,y=200)
loadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=250)


uploadButton = Button(main, text="Upload Video", command=upload)
uploadButton.place(x=50,y=300)
uploadButton.config(font=font1)

uploadButton = Button(main, text="Start Landmark Detection", command=landmarkDetection)
uploadButton.place(x=50,y=350)
uploadButton.config(font=font1)

exitButton = Button(main, text="Exit", command=exit)
exitButton.place(x=50,y=400)
exitButton.config(font=font1)

main.config(bg='chocolate1')
main.mainloop()
