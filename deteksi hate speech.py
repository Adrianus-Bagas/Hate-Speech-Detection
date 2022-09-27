import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter.scrolledtext import *
from PIL import ImageTk, Image
import tkinter.filedialog
from summarizer import Summarizer,TransformerSummarizer
import joblib
saved_model = joblib.load('model_svc.joblib') 
saved_tfidf = joblib.load('TF-IDF Vectorizer.joblib')

window = Tk()
window.title("Hate Speech Detection")
window.geometry("300x200")

img_good = ImageTk.PhotoImage(Image.open("happiness.png"))
img_bad = ImageTk.PhotoImage(Image.open("angry.png"))

def deteksi():
    tweets = [inputtxt.get(1.0, "end-1c")]
    vectorized_tweets = saved_tfidf.transform(tweets).toarray()
    input_prediction = saved_model.predict(vectorized_tweets)
    if input_prediction[0]==1:
        outputtxt.insert(END,'Ya')
    else:
        outputtxt.insert(END,'Tidak')
def clear():
    inputtxt.delete("1.0", "end")
    outputtxt.delete("1.0", "end")

label_good = Label(window, image = img_good, height=50, width=50)
label_good.place(x=10,y=50)

label_bad = Label(window, image = img_bad, height=50, width=50)
label_bad.place(x=240,y=50)

label1 = tk.Label(window, text = "Masukkan Teks",fg="black")
label1.pack()

inputtxt = tk.Text(window,
                   height = 2,
                   width = 10)
  
inputtxt.pack()

b0=Button(text="Cek", width=6,command=deteksi)
b0.pack()

label2 = tk.Label(window, text = "Apakah ini ujaran kebencian ?",fg="black")
label2.pack()

outputtxt = tk.Text(window,
                   height = 2,
                   width = 10)
  
outputtxt.pack()

b1=Button(text="Reset", width=6,command=clear)
b1.pack()

window.mainloop()