import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
import shutil
import os
import sys
from PIL import Image, ImageTk

global fileName

window = tk.Tk()
window1 = tk.Tk()
window3 = tk.Toplevel();

window.title("Mango Leaf Disease Detector")

window.geometry("500x510")
#window.attributes('-fullscreen',True)
window.configure(background ="lightgreen")

title = tk.Label(text="Choose how to load picture for testing disease...", background = "lightgreen", fg="Brown", font=("", 15))
title.place(relx=0.5, rely=0.2, anchor=tk.CENTER)
load_message = tk.Label(window3,text='Note: Analyzing image may take a while', background="lightgreen",
	    fg="Brown", font=("", 12))

window1.withdraw()
window3.withdraw()

def anth():
    #window.destroy()
    window.withdraw()

    window1.title("Mango Leaf Disease Detector")
	
    window1.geometry("500x510")
    #window1.attributes('-fullscreen',True)
    window1.configure(background="lightgreen")

    def exit():
        window1.destroy()
        #window2.deiconify()
    rem = "The remedies for Anthracnose are:\n\n "
    remedies = tk.Label(window1, text=rem, background="lightgreen",
                      fg="Brown", font=("", 15))
    remedies.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
    rem1 = " 1.	Disease leaves, flowers, twigs and fruits lying on the floor of the orchard should be collected \n and all infected twigs from the tree should be pruned and burnt.. \n  2. Blossom infection can be controlled effectively by two to three sprays \n of contact or systemic fungicides during spring season at 12-15 days interval. \n  3. Best controlled by a combination of preventive measures, field fungicide sprays, and postharvest treatment."
    remedies1 = tk.Label(window1, text=rem1, background="lightgreen",
                        fg="Black", font=("", 12))
    remedies1.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    button = tk.Button(window1, text="Close this Window", command=exit)
    button.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

    window1.mainloop()


def mild():
    #window.destroy()
    window.withdraw()
    window1 = tk.Tk()

    window1.title("Mango Leaf Disease Detector")
	
    window1.geometry("500x510")
    #window1.attributes('-fullscreen',True)
    window1.configure(background="lightgreen")

    def exit():
        window1.destroy()
        
    rem = "The remedies for Powdery Mildew are:\n\n "
    remedies = tk.Label(window1, text=rem, background="lightgreen",
                      fg="Brown", font=("", 15))
    remedies.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
    rem1 = " 1.	Pruning of diseased leaves and malformed panicles reduces primary inoculums. \n  2.	3 sprays of systemic fungicides during flowering season are recommended at 12-15 days  intervals. 1st spray is recommended when there is 25% flowers opening."
    remedies1 = tk.Label(window1, text=rem1, background="lightgreen",
                         fg="Black", font=("", 12))
    remedies1.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    button = tk.Button(window1, text="Exit", command=exit)
    button.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

    window1.mainloop()

def latebl():
    #window.destroy()
    window.withdraw()
    window1 = tk.Tk()

    window1.title("Mango Leaf Disease Detector")

    window1.geometry("500x510")
	#window1.attributes('-fullscreen',True)
    window1.configure(background="lightgreen")

    def exit():
        window1.destroy()
        window3.deiconify()
    rem = "The remedies for Late Blight are:\n\n "
    remedies = tk.Label(window1, text=rem, background="lightgreen",
                      fg="Brown", font=("", 15))
    remedies.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

    rem1 = " 1. Monitor the field, remove and destroy infected leaves. \n  2. Treat organically with copper spray. \n  3. Use chemical fungicides,the best of which for tomatoes is chlorothalonil."
    remedies1 = tk.Label(window1, text=rem1, background="lightgreen",
                         fg="Black", font=("", 12))
    remedies1.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    button = tk.Button(window1, text="Exit", command=exit)
    button.place(relx=0.5, rely=0.6, anchor=tk.CENTER)



def analysis():
    global fileName
    import cv2  # working with, mainly resizing, images
    import numpy as np  # dealing with arrays
    import os  # dealing with directories
    from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
    from tqdm import \
        tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
    verify_dir = '/home/pi/plant-disease-detection/PlantDiseaseDetectionV1.1/testpicture'
    IMG_SIZE = 50
    LR = 1e-3
    MODEL_NAME = 'healthyvsunhealthy-{}-{}.model'.format(LR, '2conv-basic')

    def process_verify_data():
        verifying_data = []
        for img in tqdm(os.listdir(verify_dir)):
            path = os.path.join(verify_dir, img)
            img_num = img.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
        np.save('verify_data.npy', verifying_data)
        return verifying_data

    verify_data = process_verify_data()
    #verify_data = np.load('verify_data.npy')
	


    import tflearn
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression
    import tensorflow as tf
    tf.reset_default_graph()

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 128, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 4, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')

    import matplotlib.pyplot as plt

    fig = plt.figure()

    for num, data in enumerate(verify_data):

        img_num = data[1]
        img_data = data[0]

        y = fig.add_subplot(3, 4, num + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
        # model_out = model.predict([data])[0]
        model_out = model.predict([data])[0]

        if np.argmax(model_out) == 0:
            str_label = 'healthy'
        elif np.argmax(model_out) == 1:
            str_label = 'anthracnose'
        elif np.argmax(model_out) == 2:
            str_label = 'powderymildew'
        elif np.argmax(model_out) == 3:
            str_label = 'lateblight'

        if str_label =='healthy':
            status ="HEALTHY"
        else:
            status = "UNHEALTHY"
 
        disease = tk.Label(window3,text='',background="lightgreen",fg="Black",font=("",15))
        r = tk.Label(window3,text='Click below for remedies...', background="lightgreen", fg="Brown", font=("", 15))
        rH = tk.Label(window3,text='Plant is healthy', background="lightgreen", fg="Black",
                         font=("", 15))
            
        buttonAN = tk.Button(window3,text="Remedies", command=anth)	
        buttonMD = tk.Button(window3,text="Remedies", command=mild)
        buttonLB = tk.Button(window3,text="Remedies", command=latebl)
        buttonEX = tk.Button(window3,text="Exit")	
	
        def exit():
            disease.destroy()
            r.destroy()
			
            buttonAN.destroy()
            buttonMD.destroy()
            buttonLB.destroy()
            buttonEX.destroy()
            window3.withdraw()
            window.deiconify()
		
        #load_message.withdraw()
        message = tk.Label(window3,text='Status: '+status, background="lightgreen",
                           fg="Brown", font=("", 15))
        message.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        if str_label == 'anthracnose':
            diseasename = "Anthracnose "
            disease['text'] = "Disease Name: " +diseasename
            disease.place(relx=0.5, rely=0.6, anchor=tk.CENTER)
            r.place(relx=0.5, rely=0.7, anchor=tk.CENTER)
            buttonAN.place(relx=0.4, rely=0.8, anchor=tk.CENTER)
            buttonEX.place(relx=0.6, rely=0.8, anchor=tk.CENTER)
            buttonEX['command'] = exit
        elif str_label == 'powderymildew':
            diseasename = "Powdery Mildew "
            disease['text'] = "Disease Name: " +diseasename
            disease.place(relx=0.5, rely=0.6, anchor=tk.CENTER)
            r.place(relx=0.5, rely=0.7, anchor=tk.CENTER)
            buttonMD.place(relx=0.4, rely=0.8, anchor=tk.CENTER)
            buttonEX.place(relx=0.6, rely=0.8, anchor=tk.CENTER)
            buttonEX['command'] = exit
        elif str_label == 'lateblight':
            diseasename = "Late Blight "
            disease['text'] = "Disease Name: " +diseasename
            disease.place(relx=0.5, rely=0.6, anchor=tk.CENTER)
            r.place(relx=0.5, rely=0.7, anchor=tk.CENTER)
            buttonLB.place(relx=0.4, rely=0.8, anchor=tk.CENTER)
            buttonEX.place(relx=0.6, rely=0.8, anchor=tk.CENTER)
            buttonEX['command'] = exit
        else:
            rH.place(relx=0.5, rely=0.7, anchor=tk.CENTER)
            buttonEX.place(relx=0.5, rely=0.8, anchor=tk.CENTER)
            buttonEX['command'] = exit
	
             
def openphoto():
    global fileName
    window.withdraw()
    window3.title("Mango Leaf Disease Detector")
    window3.geometry("500x510")
    window3.configure(background="lightgreen")
    x = (window3.winfo_screenwidth() - window3.winfo_reqwidth()) / 2
    y = (window3.winfo_screenheight() - window3.winfo_reqheight()) / 2
    window3.geometry("+%d+%d" % (x, y))
    window3.deiconify();
    dirPath = "/home/pi/plant-disease-detection/PlantDiseaseDetectionV1.1/testpicture"
    fileList = os.listdir(dirPath)
    for fName in fileList:
        os.remove(dirPath + "/" + fName)
    # C:/Users/sagpa/Downloads/images is the location of the image which you want to test..... you can change it according to the image location you have  
    fName = askopenfilename(initialdir='/home/pi/plant-disease-detection/PlantDiseaseDetectionV1.1/test', title='Select image for analysis ',
                           filetypes=[('image files', '.jpg')])
    dst = "/home/pi/plant-disease-detection/PlantDiseaseDetectionV1.1/testpicture"
    shutil.copy(fName, dst)
    load = Image.open(fName)
    fileName = fName
    print("Image File:" +fileName)
    render = ImageTk.PhotoImage(load)
    img = tk.Label(window3,image=render, height="250", width="500")
    img.image = render
    img.place(x=0, y=0)
    img.place(relx=0.5, rely=0.3, anchor=tk.CENTER)
    load_message.place(relx=0.5, rely=0.65, anchor=tk.CENTER)

    button3 = tk.Button(window3,text="Analyze Image", command=analysis)
    button3.place(relx=0.5, rely=0.6, anchor=tk.CENTER)
	
    
	
def capturephoto():
    window.withdraw()
    window3.title("Mango Leaf Disease Detector")
    window3.geometry("500x510")
    window3.configure(background="lightgreen")
    x = (window3.winfo_screenwidth() - window3.winfo_reqwidth()) / 2
    y = (window3.winfo_screenheight() - window3.winfo_reqheight()) / 2
    window3.geometry("+%d+%d" % (x, y))
    window3.deiconify();
    import cv2
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Camera")
    while True:
       ret,frame = cam.read()
       cv2.imshow("Camera",frame)
       if not ret:
          break
          cv2.destroyAllWindows()
       k = cv2.waitKey(0)
       if k%256 == 27:
          break
          cv2.destroyAllWindows()
       elif k%256 == 32:
          print("Image captured!")
          img_name = "mango-leaf.jpg"
          cv2.imwrite('/home/pi/plant-disease-detection/PlantDiseaseDetectionV1.1/test/'+img_name,frame)     
          cam.release()
          cv2.destroyAllWindows()
          fileName = '/home/pi/plant-disease-detection/PlantDiseaseDetectionV1.1/test/mango-leaf.jpg'
          dst = "/home/pi/plant-disease-detection/PlantDiseaseDetectionV1.1/testpicture"
          shutil.copy(fileName, dst)
          load = Image.open(fileName)
          render = ImageTk.PhotoImage(load)
          img = tk.Label(window3,image=render, height="250", width="500")
          img.image = render
          img.place(x=0, y=0)
          img.place(relx=0.5, rely=0.3, anchor=tk.CENTER)
          load_message.place(relx=0.5, rely=0.65, anchor=tk.CENTER)
          button3 = tk.Button(window3,text="Analyze Image", command=analysis)
          button3.place(relx=0.5, rely=0.6, anchor=tk.CENTER)


button1 = tk.Button(window, text="Select Existing Photo", command = openphoto)
button1.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
#button1.grid(column=5, row=5, padx=10, pady = 10)
button2 = tk.Button(window, text="Use Camera (Press Space to take Picture)", command = capturephoto)
button2.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
#button2.grid(column=5, row=6, padx=10, pady = 10)
#text_box = tk.Text(state=tk.DISABLED)
#text_box.grid(column=0, row=3)
#sys.stdout = StdRedirector(text_box)
#sys.stderr = StdRedirector(text_box)

window.withdraw()
window.update_idletasks()  # Update "requested size" from geometry manager

x = (window.winfo_screenwidth() - window.winfo_reqwidth()) / 2
y = (window.winfo_screenheight() - window.winfo_reqheight()) / 2
window.geometry("+%d+%d" % (x, y))

# This seems to draw the window frame immediately, so only call deiconify()
# after setting correct window position
window.deiconify()

def on_closing():
    if tk.messagebox.askokcancel("Quit", "Do you want to quit?"):
        window.destroy()
        window1.destroy()
        #window3.destroy()

window.protocol("WM_DELETE_WINDOW", on_closing)
window.mainloop()






