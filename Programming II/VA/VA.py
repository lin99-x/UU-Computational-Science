'''
Name: Jinglin Gao
Date: 27/10/2022
'''

import tkinter as tk
import math
import random


class Move_Ball:
    
    def __init__(self, canvas, xposition, yposition, diameter, xvelocity, yvelocity):
        self.canvas = canvas
        self.image = canvas.create_oval(xposition, yposition, xposition + diameter, yposition + diameter, fill="black", tag="move_ball")
        self.speed = math.sqrt(xvelocity**2 + yvelocity**2)
        self.xvelocity = xvelocity
        self.yvelocity = yvelocity
        self.coordinates = [xposition, yposition]

        
    def move(self):
        coordinates = self.canvas.coords(self.image)
        self.coordinates = [coordinates[0], coordinates[1]]
        # print(self.coordinates)
        # check collision between ball and wall
        if (coordinates[2] >= (self.canvas.winfo_width()) or coordinates[0] < 0):
            self.xvelocity = -self.xvelocity
        if (coordinates[3] >= (self.canvas.winfo_height()) or coordinates[1] < 0):
            self.yvelocity = -self.yvelocity
        self.canvas.move(self.image, self.xvelocity, self.yvelocity)
        self.canvas.after(int(self.speed), self.move)

   

class Still_Ball:
    
    def __init__(self):
        self.numbers = int(entry_num.get())
        self.coordinates = []
        self.ball = []
        
        for i in range (0, int(entry_num.get())):
            x = random.randint(0, (WIDTH / SPACE_SIZE)-1) * SPACE_SIZE
            y = random.randint(0, (HEIGHT / SPACE_SIZE)-1) * SPACE_SIZE
            self.coordinates.append([x, y])

        j = 0    
        for x, y in self.coordinates:
            ball = canvas.create_oval(x, y, x + SPACE_SIZE, y + SPACE_SIZE, fill='pink', tag=j)
            j += 1
            self.ball.append(ball)
        


def start_game():
    # generate balls based on user input quantity
    # one ball is the move ball to be controlled and eat others
    xspeed = float(entry_xspd.get())
    yspeed = float(entry_yspd.get())
    mball = Move_Ball(canvas, 0, 0, SPACE_SIZE, xspeed, yspeed)
    mball.move()
    sball = Still_Ball()
    check_collision(mball, sball)
    '''
    There is still problem with the collision.
    '''

def check_collision(move_ball, still_ball):
    x, y = move_ball.coordinates
    # center of moving ball
    x0 = x + SPACE_SIZE / 2
    y0 = y + SPACE_SIZE / 2
    num = len(still_ball.coordinates)
    dis = []
    rec = []
    # k = 0
    # for x1, y1 in still_ball.coordinates:
    #     # center of still ball
    #     x10 = x1 + SPACE_SIZE / 2
    #     y10 = y1 + SPACE_SIZE / 2
    #     dist = math.sqrt((x0-x10)**2 + (y0-y10)**2)
    #     if dist <= SPACE_SIZE:
    #         del still_ball.coordinates[k]
    #         canvas.delete(k)
    #         del still_ball.ball[k]
    #     else:
    #         k += 1
    #     dis.append(distance)
    for k in range (num):
        x1, y1 = still_ball.coordinates[k]
        x10 = x1 + SPACE_SIZE / 2
        y10 = y1 + SPACE_SIZE / 2
        dist = math.sqrt((x0-x10)**2 + (y0-y10)**2)
        dis.append(dist)
    for n1 in range (len(dis)):
        if dis[n1] <= SPACE_SIZE:
            rec.append(n1)
    for j in rec:
        global score
        score += 1
        label.config(text="Score:{}".format(score))
        print(still_ball.coordinates)
        del still_ball.coordinates[j]
        canvas.delete(j)           
        del still_ball.ball[j]
    window.after(int(move_ball.speed), check_collision, move_ball, still_ball)            
        

def resume_game():
    canvas.delete('all')
    entry_num.delete(0, 'end')
    entry_xspd.delete(0, 'end')
    entry_yspd.delete(0, 'end')
    global score
    score = 0
    label.config(text="Score:{}".format(score))
            
def game_over():
    canvas.delete('all')
    canvas.create_text(canvas.winfo_width()/2, canvas.winfo_height()/2, 
                       font=('consolas', 70), text="GAME OVER", fill="red", tag="gameover")
    
window = tk.Tk()
window.title("Eat Game")

WIDTH = 700
HEIGHT = 700
SPACE_SIZE = 20
score = 0

window.rowconfigure(1, minsize=HEIGHT, weight=1)
window.columnconfigure(2, minsize=WIDTH, weight=1)

canvas = tk.Canvas(window, bg="#FFFFA5")
label = tk.Label(window, text="Score:{}".format(score), font=('consoles', 40), fg="red", bg="black")

frm_buttons = tk.Frame(window, relief=tk.RAISED, bd=4)
la_num = tk.Label(frm_buttons, text="Amount: ")
entry_num = tk.Entry(frm_buttons, width = 10)
la_xspd = tk.Label(frm_buttons, text="XSpeed: ")
entry_xspd = tk.Entry(frm_buttons, width = 10)
la_yspd = tk.Label(frm_buttons, text="YSpeed: ")
entry_yspd = tk.Entry(frm_buttons, width = 10)
btn_start = tk.Button(frm_buttons, text="Start", bg="white", command=start_game)
btn_pause = tk.Button(frm_buttons, text="Pause", bg="white")
btn_clear = tk.Button(frm_buttons, text="Clear", bg="white", command=resume_game)
btn_stop = tk.Button(frm_buttons, text="Stop", fg="red", bg="white", command=game_over)

# layout
label.grid(row=0, column=2, sticky="ew")
la_num.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
entry_num.grid(row=1, column=1, sticky="ew",padx=5)
la_xspd.grid(row=2, column=0, sticky="ew", padx=5)
entry_xspd.grid(row=2, column=1, sticky="ew", padx=5)
la_yspd.grid(row=3, column=0, sticky="ew", padx=5)
entry_yspd.grid(row=3, column=1, sticky="ew", padx=5)
btn_start.grid(row=4, column=1, sticky="ew", padx=10, pady=5)
btn_pause.grid(row=5, column=1, sticky="ew", padx=10, pady=5)
btn_clear.grid(row=6, column=1, sticky="ew", padx=10, pady=5)
btn_stop.grid(row=7, column=1, sticky="ew", padx=10, pady=5)


frm_buttons.grid(row=1, column=1, sticky="ns")
canvas.grid(row=1, column=2, sticky="nsew")


window.mainloop()