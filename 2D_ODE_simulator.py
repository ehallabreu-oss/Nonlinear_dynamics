import numpy as np
import pygame
import pygame_widgets

# ---colour palette---

yellow = (255, 211, 25)
green_1 = (75, 147, 149)
green_2 = (33, 201, 37)
turquoise = (63, 191, 175)
blue = (3, 94, 232)
light_blue = (45, 226, 230)
pink = 	(246, 1, 157)
red = (222, 0, 80)
purple = (151, 0, 204)
white = (220, 220, 220)
black = (23, 12, 30)
grey = (100, 100, 100)
bright_grey = (160, 160, 160)

# ---definitions---

# nullclines
def nullc(x, a, b):
    return -a/b * x

# linear transformation
def transform(x, v, params):
    a, b, c, d = params
    return np.array([a*x + b*v, c*x + d*v])

# eigen values and slopes
def get_eigen_values(params):
    a, b, c, d = params
    eigen_1 = (a+d + np.emath.sqrt((a+d)**2 - 4*(a*d - b*c)))/2
    eigen_2 = (a+d - np.emath.sqrt((a+d)**2 - 4*(a*d - b*c)))/2
    return [eigen_1, eigen_2]

def get_eigen_y(x, eigen_value, params):
    """
    computes the slope of each eigen vector
    returns the y value given an input x and the eigen value
    """
    a, b, c, d = params
    if b != 0:
        slope = (eigen_value - a)/b  
    elif eigen_value - d != 0:
        slope = c/(eigen_value - d) 
    
    return  slope * x

def get_eigen_x(y, eigen_value, params):
    """
    computes the slope of each eigen vector
    returns the x value given an input y and the eigen value
    """
    a, b, c, d = params
    if eigen_value - a != 0:
        slope = b/(eigen_value - a) 
    elif c != 0:
        slope = (eigen_value - d)/c
    
    return  slope * y
  
# button logic
def button_press(mouse_x, mouse_y, rect, action):
    if mouse_x > rect.left and mouse_x < rect.right and mouse_y > rect.top and mouse_y < rect.bottom:
        return not action
    return action

# ode solver
def RK4_update(x, v, dt, params):
    a, b, c, d = params

    dot = np.array([x, v])

    def f(dot):
        x, v = dot     
        return np.array([a*x + b*v, c*x + d*v])
    
    k1 = f(dot)-1
    k2 = f(dot + dt/2 * k1)
    k3 = f(dot + dt/2 * k2) 
    k4 = f(dot + dt * k3)  

    dot += dt/6 * (k1 + 2*k2 + 2*k3 + k4) 
    x, v = float(dot[0]), float(dot[1]) 

    return x, v

def draw_arrow_head(xs_end, xs_start, vs_end, vs_start, arrow_size, colour, screen):
    dx = xs_end - xs_start
    dy = vs_end - vs_start
    length = np.sqrt(dx**2 + dy**2)

    if length != 0.0:
        px = -dy/length
        py = dx/length

        back_x = xs_end - 0.4*dx
        back_y = vs_end - 0.4*dy

        arrow_right = (back_x + arrow_size * px, back_y + arrow_size * py)
        arrow_left = (back_x - arrow_size * px, back_y - arrow_size * py)  

        pygame.draw.polygon(screen, colour, [(xs_end, vs_end), arrow_right, arrow_left])



# ---main loop---

def main():
    pygame.init()
    
    width, height = 800, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Linear ODE visualiser')
    clock = pygame.time.Clock()

    # ----buttons---- 
    font = pygame.font.Font('freesansbold.ttf', 17)
    margin = 30
    
    # x(t)
    xtext = font.render('[Show x(t)]', True, light_blue, black)
    xRect = xtext.get_rect()
    pos_x, pos_y = 700, 100
    xRect.center = (pos_x, pos_y)
    
    # v(t)
    vtext = font.render('[Show v(t)]', True, green_1, black)
    vRect = vtext.get_rect()
    vRect.center = (pos_x, pos_y + margin)

    # nullclines
    ntext = font.render('[Nullclines]', True, pink, black)
    nRect = ntext.get_rect()
    nRect.center = (pos_x, pos_y + 2*margin)

    # eigen vectors
    etext = font.render('[Eigen Vectors]', True, red, black)
    eRect = etext.get_rect()
    eRect.center = (pos_x, pos_y + 3*margin)

    # grid lines for linear transform
    gtext = font.render('[Transform]', True, green_2, black)
    gRect = gtext.get_rect()
    gRect.center = (pos_x, pos_y + 4*margin)

    # animate linear transfomr
    atext = font.render('[Animate]', True, white, black)
    aRect = atext.get_rect()
    aRect.center = (pos_x, pos_y + 5*margin)

    # ----sliders----
    low_bound = 70
    

    # Math range
    xmin, xmax = -50, 50
    ymin, ymax = -50, 50

    # coordinate transforms
    def x_screen_to_math(x, width=width, xmin=xmin, xmax=xmax):
        return xmin + (x/ width) * (xmax - xmin)

    def x_math_to_screen(x, width=width, xmin=xmin, xmax=xmax):
        len = xmax - xmin
        return ((x + len/2) / len) * width

    def y_screen_to_math(y, height=height, ymin=ymin, ymax=ymax):
        return ymax - (y/ height) * (ymax - ymin)

    def y_math_to_screen(y, height=height, ymin=ymin, ymax=ymax):
        len = ymax - ymin
        return (height - ((y + len/2) / len) * height)

    def x_time_to_screen(t_current, y0, t_min, t_max, height=height):
        bottom_panel = height - y0
        return height - (t_current - t_min)/(t_max - t_min) * bottom_panel 

    def v_time_to_screen(t_current, x0, t_min, t_max, width=width):
        right_panel = width - x0
        return (t_current - t_min)/(t_max - t_min) * right_panel


    # intialising trajectory to draw later
    trajectory = []
    
    # time domain
    t_current = 0.0
    time_window = 10
    x_time_history = []
    v_time_history = []

    # Simulation params
    x = 0.0
    v = 0.0
    dt = 0.01

    a, b, c, d = 1,0,0,1
    
    running = False
    xt_draw = False
    vt_draw = False
    nullcline_draw = False
    animate_transform = False
    vector_draw = True
    grid_draw = True
    eigen_draw = False
    dragging_i = False
    dragging_j = False
    
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
                if event.key == pygame.K_x:
                    xt_draw = not xt_draw
                if event.key == pygame.K_v:
                    vt_draw = not vt_draw
                if event.key == pygame.K_t:
                    animate_transform = not animate_transform
                if event.key == pygame.K_l:
                    grid_draw = not grid_draw
                if event.key == pygame.K_f:
                    vector_draw = not vector_draw

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
        
                # buttons
                xt_draw = button_press(mouse_x, mouse_y, xRect, xt_draw)
                vt_draw = button_press(mouse_x, mouse_y, vRect, vt_draw)
                nullcline_draw = button_press(mouse_x, mouse_y, nRect, nullcline_draw)
                grid_draw = button_press(mouse_x, mouse_y, gRect, grid_draw)
                animate_transform = button_press(mouse_x, mouse_y, aRect, animate_transform)
                eigen_draw = button_press(mouse_x, mouse_y, eRect, eigen_draw)
                
                # setting initial conditions
                if not running:
                    x = x_screen_to_math(mouse_x)
                    v = y_screen_to_math(mouse_y)
                    trajectory = []

                #  changing basis vectors 
                i_distance = np.sqrt((mouse_x - ix)**2 + (mouse_y - iy)**2)
                j_distance = np.sqrt((mouse_x - jx)**2 + (mouse_y - jy)**2)

                radius = 10
                if i_distance <= radius:
                    dragging_i = True

                if j_distance <= radius:
                    dragging_j = True 

            elif event.type == pygame.MOUSEBUTTONUP:
                dragging_i = False
                dragging_j = False

        # ---update----
        
        x0 = x_math_to_screen(0)
        y0 = y_math_to_screen(0)
        res = 5

        if dragging_i:
            mouse_x, mouse_y = event.pos
            a = x_screen_to_math(mouse_x) / res
            c = y_screen_to_math(mouse_y) / res

        if dragging_j:
            mouse_x, mouse_y = event.pos
            b = x_screen_to_math(mouse_x) / res
            d = y_screen_to_math(mouse_y) / res

        params = (a, b, c, d)

        eigen_values = get_eigen_values(params)

        # equation and solution text
    
        font_2 = pygame.font.Font('freesansbold.ttf', 20)

        equation_xtext = font_2.render(f" x' = {np.round(a,2)} · x + {np.round(b,2)} · v ", True, white, black)
        equation_xrect = equation_xtext.get_rect()
        equation_xrect.center = (120, low_bound-40)

        equation_vtext = font_2.render(f" v' = {np.round(c,2)} · x + {np.round(d,2)} · v ", True, white, black)
        equation_vrect = equation_vtext.get_rect()
        equation_vrect.center = (120, low_bound-10)

        solution_text = font.render(f"x(t) = A · exp({np.round(eigen_values[0], 2)}t) + B · exp({np.round(eigen_values[1], 2)}t)", True, light_blue, black)
        solution_rect = solution_text.get_rect()
        solution_rect.center = (width/2, 100)


        # linear transform animation
        speed = 0.005
        cutoff_1 = 0.7
        cutoff_2 = 0.01

        if animate_transform:

            if a-a_curr<cutoff_1 and b-b_curr<cutoff_1 and c-c_curr<cutoff_1 and d-d_curr<cutoff_1:
                speed = 0.05

            a_curr += speed * (a - a_curr)
            b_curr += speed * (b - b_curr)
            c_curr += speed * (c - c_curr)
            d_curr += speed * (d - d_curr)

            if a-a_curr<cutoff_2 and b-b_curr<cutoff_2 and c-c_curr<cutoff_2 and d-d_curr<cutoff_2:
                animate_transform = False
            
        else:
            a_curr, b_curr, c_curr, d_curr = 1,0,0,1

        params_anim = (a_curr, b_curr, c_curr, d_curr)
    
        # dot and trace
        if running:  
            x, v = RK4_update(x, v, dt, params)
        else:
            x_time_history = []
            v_time_history = []
            
        screen_x = x_math_to_screen(x)
        screen_v = y_math_to_screen(v)
        
        trajectory.append((screen_x, screen_v))
        
        # time domain
        t_current += dt
        t_min = t_current - time_window
        t_max = t_min + time_window
        
        if xt_draw:
            x_time_history.append((x, t_current))
            x_time_screen = []

            for x, t in x_time_history:
                x_screen = x_math_to_screen(x)
                t_screen_x = x_time_to_screen(t, y0, t_min, t_max, height=height)
                x_time_screen.append((x_screen, t_screen_x))
        else:
            x_time_history = []
        
        if vt_draw:
            v_time_history.append((v, t_current))
            v_time_screen = []

            for v, t in v_time_history:    
                v_screen = y_math_to_screen(v)
                t_screen_v = v_time_to_screen(t, x0, t_min, t_max, width=width)
                v_time_screen.append((t_screen_v, v_screen))
        else:
            v_time_history = []

        # reset if values explode  
        if not isinstance(x, (int, float)) or x > np.exp(10) or not isinstance(v, (int, float)) or v > np.exp(10):
            x = 0.0
            v = 0.0
            running = False 

        # ----draw-----
        screen.fill(black)

        x_vec = np.arange(xmin, xmax+res, res, dtype='float64')
        v_vec = np.arange(ymin, ymax+res, res, dtype='float64')

        X, V = np.meshgrid(x_vec, v_vec)

        # linear transform animation logic
        if animate_transform:
            field = transform(X, V, params_anim) # animate
        else:
            field = transform(X, V, params) # don't animate


        # grid lines for transform visualiation
        if grid_draw:

            # vertical lines 
            start_v = field[:,0,:]
            stop_v = field[:,-1,:]

            for i in range(len(x_vec)):
                
                v_begin = (x_math_to_screen(start_v[0,i]), 
                        y_math_to_screen(start_v[1,i]))
                
                v_end = (x_math_to_screen(stop_v[0,i]), 
                        y_math_to_screen(stop_v[1,i]))
                
                pygame.draw.line(screen, grey, v_begin, v_end, 2)

            # horizontal lines
            start_h = field[:,:,0] 
            stop_h = field[:,:,-1] 

            for j in range(len(v_vec)):
                
                h_begin = (x_math_to_screen(start_h[0,j]), 
                        y_math_to_screen(start_h[1,j]))
                
                h_end = (x_math_to_screen(stop_h[0,j]), 
                        y_math_to_screen(stop_h[1,j]))
                
                pygame.draw.line(screen, grey, h_begin, h_end, 2)

        
        # vector field
        scale = 0.05    # this is just the size of dt
        arrow_size = 2.5

        if vector_draw:
            for x_start in x_vec:
                for v_start in v_vec:
                    if animate_transform:
                        x_end, v_end = RK4_update(x_start, v_start, scale, params_anim)
                    else:
                        x_end, v_end = RK4_update(x_start, v_start, scale, params)
                    
                    xs_start = x_math_to_screen(x_start)
                    vs_start = y_math_to_screen(v_start)

                    xs_end = x_math_to_screen(x_end)
                    vs_end = y_math_to_screen(v_end)

                    pygame.draw.line(screen, purple, (xs_start, vs_start), (xs_end, vs_end), 2)
                    
                    # arrow head
                    draw_arrow_head(xs_end, xs_start, vs_end, vs_start, arrow_size, purple, screen)


        # draw basis vectors

        def draw_basis(hor, ver, colour, x0=x0, y0=y0):  
            x = x_math_to_screen(hor*res)
            y = y_math_to_screen(ver*res)
            pygame.draw.line(screen, colour, (x0, y0), (x, y), 2)
            draw_arrow_head(x, x0, y, y0, arrow_size, colour, screen)

        if animate_transform:
            draw_basis(a_curr, c_curr, green_2) # i hat
            draw_basis(b_curr, d_curr, red) # j hat
        else:
            draw_basis(a, c, green_2) # i hat
            draw_basis(b, d, red) # j hat


        ix = x_math_to_screen(a*res)
        iy = y_math_to_screen(c*res)

        jx = x_math_to_screen(b*res)
        jy = y_math_to_screen(d*res)
                

        # nullclines
        start = x_screen_to_math(0)
        end = x_screen_to_math(width)
        
        if nullcline_draw:
            
            if b != 0:
                pygame.draw.line(screen, pink, (0, y_math_to_screen(nullc(start,a,b), height, ymin, ymax)),
                                (width, y_math_to_screen(nullc(end,a,b), height, ymin, ymax)), 2)
            else:
                pygame.draw.line(screen, pink, (x0, height) , (x0, 0), 2)

            if d != 0:
                pygame.draw.line(screen, pink, (0, y_math_to_screen(nullc(start,c,d), height, ymin, ymax)),
                                (width, y_math_to_screen(nullc(end,c,d), height, ymin, ymax)), 2)
            else:
                pygame.draw.line(screen, pink, (x0, height) , (x0, 0), 2)

        # eigen vectors        
        count = 0
        if eigen_draw:
            for eigen in eigen_values:
                if isinstance(eigen, float) and (b != 0 or eigen - d != 0):
                    # y values for start (s) and end (e) positions of both eigen vectors
                    eigen_ys = get_eigen_y(start, eigen, params)
                    eigen_ye = get_eigen_y(end, eigen, params)
                    
                    # converting them to screen coordinates
                    start_y = y_math_to_screen(eigen_ys, height, ymin, ymax)
                    end_y = y_math_to_screen(eigen_ye, height, ymin, ymax)
                    
                    pygame.draw.line(screen, red, (0,start_y), (width, end_y), 3)

                    #label
                    count += 1
                    if count == 1:
                        label_height = height/2 + 25
                    else:
                        label_height = height/2 - 25

                    label_pos = y_screen_to_math(label_height, height, ymin, ymax)

                    if eigen - a != 0 or c != 0:
                        label_x = get_eigen_x(label_pos, eigen, params)
                        
                        screen_label_x = x_math_to_screen(label_x)
                        
                        eigen_text = font.render(f" λ = {np.round(eigen, 2)}", True, red, black)
                        eigen_rect = eigen_text.get_rect()
                        eigen_rect.center = (screen_label_x, label_height)

                        screen.blit(eigen_text, eigen_rect)


        # moving dot
        pygame.draw.circle(screen, yellow, (screen_x, screen_v), 5)

        # trace
        if len(trajectory) > 1:
            pygame.draw.lines(screen, yellow, False, trajectory, 2)

        # x time domain
        if xt_draw and len(x_time_screen) > 1:
            pygame.draw.line(screen, bright_grey, (screen_x, screen_v), (screen_x, y0), 2)
            pygame.draw.lines(screen, light_blue, False, x_time_screen, 2)

        # v time domain
        if vt_draw and len(v_time_screen) > 1:
            pygame.draw.line(screen, bright_grey, (screen_x, screen_v), (x0, screen_v), 2)
            pygame.draw.lines(screen, green_1, False, v_time_screen, 2)

        # equation and solution text
        screen.blit(equation_xtext, equation_xrect)
        screen.blit(equation_vtext, equation_vrect)
        if xt_draw:
            screen.blit(solution_text, solution_rect)

        # buttons
        screen.blit(xtext, xRect)
        screen.blit(vtext, vRect)
        screen.blit(ntext, nRect)
        screen.blit(etext, eRect)
        screen.blit(gtext, gRect)
        if grid_draw:
            screen.blit(atext, aRect)

        pygame_widgets.update(events)
        pygame.display.flip()

if __name__ == "__main__":
    main()