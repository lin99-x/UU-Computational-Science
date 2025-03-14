import tkinter as tk
import math
import random


ball_coordslist={}
ball_list=[]

class Move_Ball:
    # always start from left up corner
    def __init__(self):
        self.ball = []
        self.coordinates = [[0, 0]]
        ball = canvas.create_oval(0, 0, SPACE_SIZE, SPACE_SIZE, fill='black', tag='move_ball')
        self.ball.append(ball)
 

class Still_Ball:
    
    def __init__(self, tg):
        x = random.randint(0, (WIDTH / SPACE_SIZE)-1) * SPACE_SIZE
        y = random.randint(0, (HEIGHT / SPACE_SIZE)-1) * SPACE_SIZE
        
        self.coordinates = [x, y]
        canvas.create_oval(x, y, x + SPACE_SIZE, y + SPACE_SIZE, fill="pink", tag=tg)


    
def turn(move_ball):
    [x, y] = move_ball.coordinates[0]
    if direction == 'up':
        y -= SPACE_SIZE
    elif direction == 'down':
        y += SPACE_SIZE
    elif direction == 'left':
        x -= SPACE_SIZE
    elif direction == 'right':
        x += SPACE_SIZE
    
    move_ball.coordinates[0] = [x, y]
    ball = canvas.create_oval(x, y, x + SPACE_SIZE, y + SPACE_SIZE, fill='black')
    move_ball.ball.append(ball)
    canvas.delete(move_ball.ball[0])
    del move_ball.ball[0]
    for coords in ball_coordslist.values():
        print(coords)
        if move_ball.coordinates == coords:
            ball_no = {i for i in ball_coordslist if ball_coordslist[i] == coords}
            canvas.delete(ball_no)
            del ball_list[ball_no]
            del ball_coordslist[ball_no]
    
    if ball_list == []:
        game_over()
    else:
        window.after(int(entry_xspd.get()), turn, move_ball)
        
def check_collisions(move_ball):
    x, y = move_ball.coordinates[0]
    if x < 0 or x > WIDTH:
        x *= (-1)
    elif y < 0 or y > HEIGHT:
        y *= (-1)
            
def change_direction(new_direction):
    
    global direction
    direction = new_direction

def start_game():
    # generate balls based on user input quantity
    # one ball is the main ball to be controlled and eat others and it will get bigger
    num = int(entry_num.get())
    xspeed = float(entry_xspd.get())
    yspeed = float(entry_yspd.get())
    mball = Move_Ball()   
    for i in range (num):
        ball = Still_Ball(i)
        ball_list.append(ball)
        ball_coordslist[i] = ball.coordinates
    turn(mball)
    print(11)

    # check_collision(mball)
    '''
    There is still problem with the collision.
    '''

# def check_collision(move_ball):
#     x, y = move_ball.coordinates
#     num = len(ball_coordslist)
#     dis = []
#     rec = []
#     for n in range (num):
#         x1, y1 = ball_coordslist[n]
#         distance = math.sqrt((x-x1)**2 + (y-y1)**2)
#         dis.append(distance)
#     for n1 in range (len(dis)):
#         if dis[n1] <= SPACE_SIZE:
#             rec.append(n1)
#     for j in rec:
#         print(j)
#         del ball_coordslist[j]
#         canvas.delete(j)           
#         del ball_list[j]
#     window.after(int(move_ball.speed), check_collision, move_ball)            
        

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
direction = 'down'

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
btn_initial = tk.Button(frm_buttons, text="Initialization", bg="white")
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
btn_initial.grid(row=4, column=1, sticky="ew", padx=10, pady=5)
btn_start.grid(row=5, column=1, sticky="ew", padx=10, pady=5)
btn_pause.grid(row=6, column=1, sticky="ew", padx=10, pady=5)
btn_clear.grid(row=7, column=1, sticky="ew", padx=10, pady=5)
btn_stop.grid(row=8, column=1, sticky="ew", padx=10, pady=5)


frm_buttons.grid(row=1, column=1, sticky="ns")
canvas.grid(row=1, column=2, sticky="nsew")

window.bind('<Left>', lambda event: change_direction('left'))
window.bind('<Right>', lambda event: change_direction('right'))
window.bind('<Up>', lambda event: change_direction('up'))
window.bind('<Down>', lambda event: change_direction('down'))



window.mainloop()