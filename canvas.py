from tkinter import *
from PIL import Image
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from utils import *

b1 = "up"
xold, yold = None, None
b3 = "up"
colors = {'draw': 'white', 'erase':'black', 'bg':'black'}
probabilities = []
progress_bars = []
drawing_area = None
def main():
    global drawing_area
    root = Tk()
    drawing_area = Canvas(root, width=500, height=500, cursor='dot', bg=colors['bg'])
    clear()
    drawing_area.pack(side=LEFT)
    buildController(root)
    drawing_area.bind("<Motion>", motion)
    drawing_area.bind("<ButtonPress-1>", b1down)
    drawing_area.bind("<ButtonRelease-1>", b1up)
    drawing_area.bind("<ButtonPress-3>", b3down)
    drawing_area.bind("<ButtonRelease-3>", b3up)
    root.mainloop()
def clear():
    global drawing_area, colors
    drawing_area.create_rectangle(0,0,510,510, fill=colors['erase'], outline=colors['erase'])

def buildController(root):
    controller = Frame(root)
    btn_clear = Button(controller, text='clear', width=20, command=lambda : clear())
    btn_classify = Button(controller, text='classify image', width=20 ,command=lambda : save_as_png(drawing_area, "image"))
    btn_map = Button(controller, text='map pixels', width=20 ,command=lambda : mapImage(drawing_area, "image"))
    controller.pack()
    btn_clear.grid(column=0, row=0, columnspan=2)
    btn_classify.grid(column=0, row=1, columnspan=2)
    btn_map.grid(column=0, row=2, columnspan=2)
    buildProgressBars(controller)

def buildProgressBars(controller):
    label = Label(controller, text='Probabilities')
    label.grid(column=0, row=3, columnspan=2)
    for x in range(0, 10):
        var_bar = DoubleVar()
        var_bar.set(0)
        probabilities.append(var_bar)
        label = Label(controller, text=str(x)+': ')
        label.grid(column=0, row=x+4)
        bar = ttk.Progressbar(controller, variable=var_bar, maximum=100, mode='determinate')
        bar.grid(column=1, row=x+4)
        progress_bars.append(bar)
        label = Label(controller, textvariable=var_bar, width=20)
        label.grid(column=2, row=x+4)
def mapImage(canvas, fileName):
    canvas.postscript(file = fileName + '.eps') 
    img = Image.open(fileName + '.eps') 
    img.thumbnail((28,28), Image.ANTIALIAS)
    img.save(fileName + '.png', 'png')
    img_array = np.array(img)[:,:, :1]
    img_array = img_array.reshape(28,28)
    plt.figure()
    plt.imshow(img_array)
    plt.colorbar()
    plt.grid(False)
    plt.show()
def save_as_png(canvas,fileName):
    canvas.postscript(file = fileName + '.eps') 
    img = Image.open(fileName + '.eps') 
    img.thumbnail((28,28), Image.ANTIALIAS)
    img.save(fileName + '.png', 'png')
    img_array = np.array(img)[:,:, :1]
    img_array = img_array.reshape(28,28)
    img_array = img_array / 255.0
    response = getValue(img_array.flatten())
    response = response.clip(0)*100
    applyValues(response)

def applyValues(vector):
    for i in range(0, 10):
        probabilities[i].set(vector[0][i])
def b3down(event):
    global b3
    b3 = "down"

def b3up(event):
    global b3, xold, yold
    b3 = "up"
    xold = None
    yold = None 
def b1down(event):
    global b1
    b1 = "down"
def b1up(event):
    global b1, xold, yold
    b1 = "up"
    xold = None
    yold = None

def motion(event):
    global xold, yold
    if b1 == "down":
        if xold is not None and yold is not None:
            event.widget.create_line(xold,yold,event.x,event.y,smooth=TRUE, width=30, capstyle=ROUND, fill=colors['draw'])
        xold = event.x
        yold = event.y
    elif b3 == "down":
        if xold is not None and yold is not None:
            event.widget.create_line(xold,yold,event.x,event.y,smooth=TRUE, width=50, capstyle=ROUND, fill=colors['erase'])
        xold = event.x
        yold = event.y

if __name__ == "__main__":
    main()