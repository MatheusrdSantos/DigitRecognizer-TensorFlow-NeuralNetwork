from tkinter import *
from PIL import Image
from tkinter import ttk

b1 = "up"
xold, yold = None, None
b3 = "up"
colors = {'draw': 'black', 'erase':'white', 'bg':'white'}
probabilities = []
drawing_area = None
var_barra = None
#var_barra.set(10) # k é um número entre 0 e o máximo 
                 # (definido como 30 no exemplo acima)
def main():
    global drawing_area
    root = Tk()
    drawing_area = Canvas(root, width=500, height=500, cursor='dot', bg=colors['bg'])
    drawing_area.grid(row=0, column=0)
    buildController(root)
    drawing_area.bind("<Motion>", motion)
    drawing_area.bind("<ButtonPress-1>", b1down)
    drawing_area.bind("<ButtonRelease-1>", b1up)
    drawing_area.bind("<ButtonPress-3>", b3down)
    drawing_area.bind("<ButtonRelease-3>", b3up)
    root.mainloop()
def clear():
    global drawing_area, colors
    drawing_area.create_rectangle(0,0,500,500, fill=colors['erase'], outline=colors['erase'])

def buildController(root):
    controller = Frame(root, bg="red")
    btn_1 = Button(controller, text='clear', width=20, command=lambda : clear())
    btn_2 = Button(controller, text='analyse', width=20 ,command=lambda : save_as_png(drawing_area, "image"))
    controller.grid(column=1, row=0)
    btn_1.pack()
    btn_2.pack()

def buildProgressBars(controller):
    for x in range(0, 10):

    var_bar0 = DoubleVar()
    var_bar0.set(50)
    bar = ttk.Progressbar(controller, variable=var_bar0, maximum=100)
    bar.pack(fill=X)
    
    var_bar2 = DoubleVar()
    var_bar2.set(50)
    bar = ttk.Progressbar(controller, variable=var_bar2, maximum=100)
    bar.pack(fill=X)
    
def save_as_png(canvas,fileName):
    # save postscipt image 
    canvas.postscript(file = fileName + '.eps') 
    # use PIL to convert to PNG 
    img = Image.open(fileName + '.eps') 
    #img.resize((28,28), Image.ANTIALIAS)
    img.thumbnail((28,28), Image.ANTIALIAS)
    img.save(fileName + '.png', 'png') 

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
    b1 = "down"           # you only want to draw when the button is down
                          # because "Motion" events happen -all the time-

def b1up(event):
    global b1, xold, yold
    b1 = "up"
    xold = None           # reset the line when you let go of the button
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