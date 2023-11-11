import os
import turtle 

screen_1 = turtle.Screen()
screen_1.title("Ping Pong Game")
screen_1.bgcolor("red")
screen_1.setup(width = 1000, height = 1000)

left_paddle = turtle.Turtle()
left_paddle.speed(10)
left_paddle.shape("circle")
left_paddle.color("yellow")
left_paddle.shapesize(stretch_wid = 6, stretch_len = 2)
left_paddle.penup()
left_paddle.goto(-400,0)


right_paddle = turtle.Turtle()
right_paddle.speed(10)
right_paddle.shape("circle")
right_paddle.color("yellow")
right_paddle.shapesize(stretch_wid = 6, stretch_len = 2)
right_paddle.penup()
right_paddle.goto(400,0)

hit_ball = turtle.Turtle()
hit_ball.speed(250)
hit_ball.shape("circle")
hit_ball.color("Black")
hit_ball.penup()
hit_ball.goto(0,0)
hit_ball.dx = 12
hit_ball.dy = -12

left_player = 0
right_player = 0

sketch_1 = turtle.Turtle()
sketch_1.speed(0)
sketch_1.color("blue")
sketch_1.penup()
sketch_1.hideturtle()
sketch_1.goto(0,260)
sketch_1.write("Left Player: 0   Right Player: 0", align = "center", font = ("Courier", 24, "normal") )

def paddle_L_up():
    y = left_paddle.ycor()
    y += 30
    left_paddle.sety(y)

def paddle_L_down():
    y = left_paddle.ycor()
    y -= 30
    left_paddle.sety(y)

def paddle_R_up():  
    y = right_paddle.ycor()  
    y += 30  
    right_paddle.sety(y) 

def paddle_R_down():
    y = right_paddle.ycor()
    y -= 30
    right_paddle.sety(y)

screen_1.listen
screen_1.onkeypress(paddle_L_up, "Up")
screen_1.onkeypress(paddle_L_down, "Down")
screen_1.onkeypress(paddle_R_up, "w")
screen_1.onkeypress(paddle_R_down, "s")






