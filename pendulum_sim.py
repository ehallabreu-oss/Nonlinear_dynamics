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


def RK4_update(x, v, dt, k, m, g):
    dot = np.array([x, v])

    def f(dot):
        x, v = dot    # a coordinate/point   
        return np.array([v, (-k/m) * x + g])
    
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
    k_slider = Slider(screen, 550, 80, 200, 10, min= 0, max=10, step=0.01)
    m_slider = Slider(screen, 550, 100, 200, 10, min= 1, max=10, step=0.01)
    g_slider = Slider(screen, 550, 120, 200, 10, min= 0, max=10, step=0.01)

    # Math range
    xmin, xmax = -50, 50
    ymin, ymax = -50, 50

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
        k = k_slider.getValue()
        m = m_slider.getValue()
        g = g_slider.getValue()

        # dot
        if running:  
            x, v = RK4_update(x, v, dt, k, m, g)

        # reset if values explode  
        if not isinstance(x, (int, float)) or not math.isfinite(x) or x > np.exp(10) or not isinstance(v, (int, float)) or not math.isfinite(v) or v > np.exp(10):
            x = 0.0
            v = 0.0
            running = False
            screen_x = x_math_to_screen(x, width, xmin, xmax)
            screen_v = y_math_to_screen(v, height, ymin, ymax)   

        # ----draw-----
        screen.fill((30, 30, 30))

        # x and y axis
        y0 = y_math_to_screen(0, height, ymin, ymax)
        x0 = x_math_to_screen(0, width, xmin, xmax)
        pygame.draw.line(screen, (200,200,200), (0, y0) , (width, y0), 2)
        pygame.draw.line(screen, (200,200,200), (x0, height) , (x0, 0), 2)

        # moving dot
        screen_x = x_math_to_screen(x, width, xmin, xmax)
        screen_v = y_math_to_screen(v, height, ymin, ymax)
        pygame.draw.circle(screen, (255, 100, 100), (screen_x, screen_v), 8)

        # vector field
        
        # aesthetics
        res = 26
        scale = 0.1    # this is just the size of dt

        X = np.linspace(xmin, xmax, res)
        V = np.linspace(ymin, ymax, res)

        for x_start in X:
            for v_start in V:
                x_end, v_end = RK4_update(x_start, v_start, scale, k, m, g)
                
                xs_start = x_math_to_screen(x_start, width, xmin, xmax)
                vs_start = y_math_to_screen(v_start, height, ymin, ymax)

                xs_end = x_math_to_screen(x_end, width, xmin, xmax)
                vs_end = y_math_to_screen(v_end, height, ymin, ymax)

                pygame.draw.line(screen, (50, 50, 200), (xs_start, vs_start), (xs_end, vs_end), 2)
                
        pygame_widgets.update(events)
        pygame.display.flip()

if __name__ == "__main__":
    main()