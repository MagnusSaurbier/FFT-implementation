import tkinter
from DFT import Function
from DFT import fourier
# Tkinter Window
root = tkinter.Tk()
root.title("Plot Input")
width, height = 2**10, 768
#root.geometry(f"{width}x{height}")

# Canvas to draw a Function into
canvas = tkinter.Canvas(root, width=width, height=height)
canvas.pack()
analyzeButton = tkinter.Button(root, text="Analyze", command=lambda: fourier(f))
analyzeButton.pack()
resetButton = tkinter.Button(root, text="Reset", command=lambda: reset())
resetButton.pack()

# Function class
class PlotFunction(Function):
    def __init__(self, N, size):
        self.size=size
        self.points = {}
        self.N = N
        self.ovals = {}
    def reset(self):
        self.points = {}
        self.ovals = {}
    def evaluate(self, x):
        if x in self.points:
            return self.points[x]
        return self.interpolate(x)
    def addPoint(self, x, y):
        x,y = x//self.size*self.size, y//self.size*self.size
        if x in self.ovals:
            canvas.delete(self.ovals[x])
        self.ovals[x] = canvas.create_rectangle(x, y, x+self.size, y+self.size, fill="black")
        self.points[x//self.size] = (height//2-y)//self.size
    def interpolate(self, x):
        if x < 0 or x > self.N:
            return 0
        x1 = x2 = x
        while x1>0 and x1 not in self.points:
            x1 -= 1
        while x2<self.N and x2 not in self.points:
            x2 += 1
        if x1 == 0 or x2 == self.N:
            if x1 == 0:
                if x2 == self.N:
                    return 0
                return self.points[x2]
            return self.points[x1]
        y1 = self.points[x1]
        y2 = self.points[x2]
        return y1 + (y2-y1)/(x2-x1)*(x-x1)

# Draw Function using mouse movement
def draw(event, f):
    if event.x < width and event.y < height:
        f.addPoint(event.x, event.y)
def getFunction():
    canvas.bind("<B1-Motion>", lambda event: draw(event, f))
def reset():
    f.reset()
    canvas.delete("all")

# Draw Function

print("starting")
size = 2**7
f=PlotFunction(width//size, size)
getFunction()
root.mainloop()


