import numpy as np
import pygame
import math

# derivative function
def f(x):
    return x*3 + 0.3*x**2 - 3 

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

def update(dot, dt):
    k1 = f(dot)   
    k2 = f(dot + dt/2 * k1)  
    k3 = f(dot + dt/2 * k2) 
    k4 = f(dot + dt * k3)    

    return dot + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def main():
    pygame.init()
    
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('1D ODE Simulator')
    clock = pygame.time.Clock()

    # Math range
    xmin, xmax = -20, 20
    ymin, ymax = -20, 20

    # Simulation state
    dot = 0.0
    dt = 0.01
    running = False
    
    while True:
        clock.tick(60)

        # event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    running = not running

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x = event.pos[0]
                dot = x_screen_to_math(mouse_x, width, xmin, xmax)

        # update
        if running:  
            dot = update(dot, dt)
            
        if not isinstance(dot, (int, float)) or not math.isfinite(dot) or dot > np.exp(10):
            dot = 0.0
            running = False
            screen_dot = x_math_to_screen(dot, width, xmin, xmax)

        # draw
        screen.fill((30, 30, 30))

        # x axis
        axis_y = y_math_to_screen(0, height, ymin, ymax)
        pygame.draw.line(screen, (200,200,200), (0, axis_y) , (width, axis_y), 2)

        # draw derivative function
        xs = np.linspace(xmin, xmax, 1000)
        for i in range(len(xs)-1):
            start_x = x_math_to_screen(xs[i], width, xmin, xmax)
            start_y = y_math_to_screen(f(xs[i]), height, ymin, ymax)
            
            end_x = x_math_to_screen(xs[i+1], width, xmin, xmax)
            end_y = y_math_to_screen(f(xs[i+1]), height, ymin, ymax)
            
            pygame.draw.line(screen, (100,200,255), (start_x, start_y), (end_x, end_y), 2)

        # draw moving dot
        screen_dot = x_math_to_screen(dot, width, xmin, xmax)
        pygame.draw.circle(screen, (255, 100, 100), (screen_dot, axis_y), 8)
        pygame.display.flip()

if __name__ == "__main__":
    main()