import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
from PIL import ImageTk, Image
import numpy as np

from tensorflow.keras.models import load_model

# Load the trained model to classify signs
def load_traffic_classifier():
    model = load_model('traffic_classifier_model.h5')

    # Dictionary to label all traffic signs class

    classes = { 1:'Speed limit (20km/h)',
                2:'Speed limit (30km/h)', 
                3:'Speed limit (50km/h)', 
                4:'Speed limit (60km/h)', 
                5:'Speed limit (70km/h)', 
                6:'Speed limit (80km/h)', 
                7:'End of speed limit (80km/h)', 
                8:'Speed limit (100km/h)', 
                9:'Speed limit (120km/h)', 
                10:'No passing', 
                11:'No passing veh over 3.5 tons', 
                12:'Right-of-way at intersection', 
                13:'Priority road', 
                14:'Yield', 
                15:'Stop', 
                16:'No vehicles', 
                17:'Veh > 3.5 tons prohibited', 
                18:'No entry', 
                19:'General caution', 
                20:'Dangerous curve left', 
                21:'Dangerous curve right', 
                22:'Double curve', 
                23:'Bumpy road', 
                24:'Slippery road', 
                25:'Road narrows on the right', 
                26:'Road work', 
                27:'Traffic signals', 
                28:'Pedestrians', 
                29:'Children crossing', 
                30:'Bicycles crossing', 
                31:'Beware of ice/snow',
                32:'Wild animals crossing', 
                33:'End speed + passing limits', 
                34:'Turn right ahead', 
                35:'Turn left ahead', 
                36:'Ahead only', 
                37:'Go straight or right', 
                38:'Go straight or left', 
                39:'Keep right', 
                40:'Keep left', 
                41:'Roundabout mandatory', 
                42:'End of no passing', 
                43:'End no passing veh > 3.5 tons' }
    
    return model, classes

# Initialize GUI
def init_gui():
    top = tk.Tk()
    top.geometry('1000x800')
    top.title('Traffic Sign Classification')
    top.configure(background='#F0F2F5')
    
    # Custom font styles
    title_font = ('Helvetica', 24, 'bold')
    button_font = ('Helvetica', 12, 'bold')
    result_font = ('Helvetica', 14)
    label_font = ('Helvetica', 16, 'bold')

    # Main header
    header_frame = tk.Frame(top, bg='#2A4D69')
    header_frame.pack(fill=tk.X)
    
    heading = Label(header_frame, 
                  text="Traffic Sign Recognition System", 
                  pady=20,
                  font=title_font,
                  background='#2A4D69',
                  foreground='white')
    heading.pack(pady=10)

    # Image display area with border
    image_frame = tk.Frame(top, 
                          bg='white', 
                          relief=tk.GROOVE, 
                          borderwidth=3)
    image_frame.pack(pady=20, padx=40, fill=tk.BOTH, expand=True)
    
    sign_image = Label(image_frame, bg='white')
    sign_image.pack(pady=40, padx=40, fill=tk.BOTH, expand=True)

    # Result display
    result_frame = tk.Frame(top, bg='#F0F2F5')
    result_frame.pack(fill=tk.X, pady=10)
    
    label = Label(result_frame, 
                text="Upload image of a traffic sign", 
                font=label_font,
                background='#F0F2F5',
                foreground='#2A4D69')
    label.pack(pady=10)

    # Button container
    button_frame = tk.Frame(top, bg='#F0F2F5')
    button_frame.pack(side=tk.BOTTOM, pady=30)

    # Upload button
    upload = Button(button_frame, 
                  text="Upload Image", 
                  command=upload_image,
                  font=button_font,
                  bg='#4B86B4',
                  fg='white',
                  activebackground='#63A1C8',
                  activeforeground='white',
                  padx=20,
                  pady=10,
                  borderwidth=0,
                  relief=tk.FLAT)
    upload.pack(side=tk.LEFT, padx=20)

    # Exit button
    exit_app = Button(button_frame, 
                    text="Exit", 
                    command=top.destroy,
                    font=button_font,
                    bg='#E74C3C',
                    fg='white',
                    activebackground='#EC7063',
                    activeforeground='white',
                    padx=20,
                    pady=10,
                    borderwidth=0,
                    relief=tk.FLAT)
    exit_app.pack(side=tk.RIGHT, padx=20)

    # Add hover effects
    def on_enter(e):
        e.widget['bg'] = '#63A1C8' if e.widget == upload else '#EC7063'
    
    def on_leave(e):
        e.widget['bg'] = '#4B86B4' if e.widget == upload else '#E74C3C'

    upload.bind("<Enter>", on_enter)
    upload.bind("<Leave>", on_leave)
    exit_app.bind("<Enter>", on_enter)
    exit_app.bind("<Leave>", on_leave)

    return top, label, sign_image


    
def classify(file_path, model, classes, label):
    image = Image.open(file_path)
    image = image.resize((30, 30))  
    image = np.array(image)
    image = np.expand_dims(image, axis=0)

    pred_probs = model.predict(image)
    pred = np.argmax(pred_probs, axis=1)

    sign = classes[pred[0] + 1]  
    print(sign)
    label.configure(foreground='#011638', text=sign)
    

def show_classify_button(file_path, model, classes):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path, model, classes, label), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path, model, classes)
    except:
        pass

if __name__ == "__main__":
    model, classes = load_traffic_classifier()
    top, label, sign_image = init_gui()
    top.mainloop()
    
    
    

