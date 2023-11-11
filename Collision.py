while True:  
    screen_1.update()  
   
    hit_ball.setx(hit_ball.xcor() + hit_ball.dx)  
    hit_ball.sety(hit_ball.ycor() + hit_ball.dy)  
   
    # Check all the borders  
    if hit_ball.ycor() > 280:  
        hit_ball.sety(280)  
        hit_ball.dy *= -1  
   
    if hit_ball.ycor() < -280:  
        hit_ball.sety(-280)  
        hit_ball.dy *= -1  
   
    if hit_ball.xcor() > 500:  
        hit_ball.goto(0, 0)  
        hit_ball.dy *= -1  
        left_player += 1  
        sketch_1.clear()  
        sketch_1.write("Left_player : {}    Right_player: {}".format(  
                      left_player, right_player), align = "center",  
                      font = ("Courier", 24, "normal"))  
   
    if hit_ball.xcor() < -500:  
        hit_ball.goto(0, 0)  
        hit_ball.dy *= -1  
        right_player += 1  
        sketch_1.clear()  
        sketch_1.write("Left_player : {}    Right_player: {}".format(  
                                 left_player, right_player), align = "center",  
                                 font = ("Courier", 24, "normal"))  
   
    # Collision of ball and paddles  
    if (hit_ball.xcor() > 360 and  
                        hit_ball.xcor() < 370) and (hit_ball.ycor() < right_paddle.ycor() + 40 and  
                        hit_ball.ycor() > right_paddle.ycor() - 40):  
                        hit_ball.setx(360)  
                        hit_ball.dx *= -1  
          
    if (hit_ball.xcor() < -360 and  
                       hit_ball.xcor() > -370) and (hit_ball.ycor() < left_paddle.ycor() + 40 and  
                       hit_ball.ycor() > left_paddle.ycor() - 40):  
                       hit_ball.setx(-360)  
                       hit_ball.dx *= -1  