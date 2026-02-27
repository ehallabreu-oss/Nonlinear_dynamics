import numpy as np
import pygame
import pygame_widgets
from pygame_widgets.slider import Slider
import math


# coordinate transforms
def x_screen_to_math(x, width, xmin, xmax):
    return xmin + (x/ width) * (xmax - xmin)

def x_math_to_screen(x, width, xmin, xmax):
    len = xmax - xmin
    return int(((x + len/2) / len) * width)

def y_screen_to_math(y, height, ymin, ymax):
    return ymax - (y/ height) * (ymax - ymin)

def y_math_to_screen(y, height, ymin, ymax):
    len = ymax - ymin
    return int(height - ((y + len/2) / len) * height)

def x_time_to_screen(t_current, height, y0, t_min, t_max):
    bottom_panel = height - y0
    return height - (t_current - t_min)/(t_max - t_min) * bottom_panel 
        
def v_time_to_screen(t_current, width, x0, t_min, t_max):
    right_panel = width - x0
    return (t_current - t_min)/(t_max - t_min) * right_panel

# x'' = ax' + bx + c 
# x' = v
# v' = av + bx + c

def RK4_update(x, v, dt, a, b, c):
    dot = np.array([x, v])

    def f(dot):
        x, v = dot     
        return np.array([v,  a * v + b * x + c])
    
    k1 = f(dot)
    k2 = f(dot + dt/2 * k1)
    k3 = f(dot + dt/2 * k2) 
    k4 = f(dot + dt * k3)  

    dot += dt/6 * (k1 + 2*k2 + 2*k3 + k4) 
    x, v = float(dot[0]), float(dot[1]) 

    return x, v


def main():
    pygame.init()
    
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('2D ODE Pendulum Simulator')
    clock = pygame.time.Clock()

    # slider stuff
    a_slider = Slider(screen, 550, 80, 200, 10, min=-10, max=10, step=0.01)
    b_slider = Slider(screen, 550, 100, 200, 10, min=-10, max=10, step=0.01)
    c_slider = Slider(screen, 550, 120, 200, 10, min=-10, max=10, step=0.01)

    # Math range
    xmin, xmax = -50, 50
    ymin, ymax = -50, 50

    # intialising trajectory to draw later
    trajectory = []
    
    # time domain
    t_current = 0.0
    time_window = 10
    time_history = []

    # Simulation params
    x = 0.0
    v = 0.0
    dt = 0.01
    
    running = False
    
    while True:
        clock.tick(60)

        # event handling
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    running = not running

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                x = x_screen_to_math(mouse_x, width, xmin, xmax)
                v = y_screen_to_math(mouse_y, height, ymin, ymax)      

        # ---update----
        
        # sliders
        a = a_slider.getValue()
        b = b_slider.getValue()
        c = c_slider.getValue()
    
        # dot and trace
        if running:  
            x, v = RK4_update(x, v, dt, a, b, c)
        else:
            trajectory = []
            time_history = []
            
        screen_x = x_math_to_screen(x, width, xmin, xmax)
        screen_v = y_math_to_screen(v, height, ymin, ymax)

        y0 = y_math_to_screen(0, height, ymin, ymax)
        x0 = x_math_to_screen(0, width, xmin, xmax)
        
        trajectory.append((screen_x, screen_v))
        
        # time domain
        t_current += dt
        time_history.append((x, v, t_current))

        t_min = t_current - time_window
        t_max = t_min + time_window

        x_time_screen = []
        v_time_screen = []
        for x, v, t in time_history:
            x_screen = x_math_to_screen(x, width, xmin, xmax)
            v_screen = y_math_to_screen(v, height, ymin, ymax)
            
            t_screen_x = x_time_to_screen(t, height, y0, t_min, t_max) 
            t_screen_v = v_time_to_screen(t, width, x0, t_min, t_max)
            
            x_time_screen.append((x_screen, t_screen_x))  
            v_time_screen.append((t_screen_v, v_screen))     

        # reset if values explode  
        if not isinstance(x, (int, float)) or x > np.exp(10) or not isinstance(v, (int, float)) or v > np.exp(10):
            x = 0.0
            v = 0.0
            running = False 

        # ----draw-----
        screen.fill((30, 30, 30))

        # vector field
        res = 26
        scale = 0.1    # this is just the size of dt

        X = np.linspace(xmin, xmax, res)
        V = np.linspace(ymin, ymax, res)

        for x_start in X:
            for v_start in V:
                x_end, v_end = RK4_update(x_start, v_start, scale, a, b, c)
                
                xs_start = x_math_to_screen(x_start, width, xmin, xmax)
                vs_start = y_math_to_screen(v_start, height, ymin, ymax)

                xs_end = x_math_to_screen(x_end, width, xmin, xmax)
                vs_end = y_math_to_screen(v_end, height, ymin, ymax)

                pygame.draw.line(screen, (50, 50, 200), (xs_start, vs_start), (xs_end, vs_end), 2)

        # x and y axis
        pygame.draw.line(screen, (200,200,200), (0, y0) , (width, y0), 2)
        pygame.draw.line(screen, (200,200,200), (x0, height) , (x0, 0), 2)

        # moving dot
        pygame.draw.line(screen, (150, 150, 150), (screen_x, screen_v), (screen_x, y0), 2)
        pygame.draw.line(screen, (150, 150, 150), (screen_x, screen_v), (x0, screen_v), 2)
        pygame.draw.circle(screen, (255, 100, 100), (screen_x, screen_v), 8)

        # trace
        if len(trajectory) > 1:
            pygame.draw.lines(screen, (255, 100, 100), False, trajectory, 2)

        # x time domain
        if len(x_time_screen) > 1:
            pygame.draw.lines(screen, (180, 180, 180), False, x_time_screen, 2)

        # v time domain
        if len(v_time_screen) > 1:
            pygame.draw.lines(screen, (180, 180, 180), False, v_time_screen, 2)

                
        pygame_widgets.update(events)
        pygame.display.flip()

if __name__ == "__main__":
    main()