import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class spring_mass:

    name = 'Mass-Spring'
    axes_names = [r"$x \ \[m]$", r"$y \ \[m]$"]

    def __init__(self,x_0,v_0,k,m,t_0 = 0,h=0.02):
        self.y0 = np.array([x_0 , v_0])
        self.h = h             
        self.k = k
        self.m = m
        self.time_span = (t_0, t_0 + 10)
        self.time_steps = np.arange(t_0 , t_0 +10, h)
        self.solution = None

    def pendulum_fode(self,t,y):

        x , v = y
        
        dx = v
        dv = - (self.k / self.m) * x
        return np.array([dx,dv])


    def solution_method(self):
        sol = solve_ivp(fun = self.pendulum_fode,t_span = self.time_span,y0 = self.y0, t_eval= self.time_steps)
        self.solution = sol.y.T
        return self.solution


spring = spring_mass(1, 0, 2, 1)

spring_solution = spring.solution_method()

x = spring_solution[:,0]
v = spring_solution[:,1]


#Graphing

fig, axs = plt.subplots(1,2,figsize = (12,5), facecolor = "#0B0F19")

visual = axs[0]
energy = axs[1]

obj, = visual.plot([],[],'o', label='Mass', color = '#E60033', zorder = 3)
spring_line, = visual.plot([],[],'-',label='Spring',color = '#39C5BB',zorder = 2)
wall, =visual.plot([],[],'-', color = '#FFD700',zorder = 2)
visual.set_xlim(-3,np.max(x) + 0.2)
visual.set_ylim(-1, 1)
visual.xaxis.label.set_color('#E5E7EB')
visual.yaxis.label.set_color('#E5E7EB')
visual.set_aspect('equal', adjustable='box')
visual.tick_params(colors='#9CA3AF', labelsize=10)
visual.set_facecolor('#111827')
legend_visual = visual.legend(loc = 'upper right', frameon = False)
for text in legend_visual.get_texts():
    text.set_color('#FFFFFF')
visual.grid(color = '#374151')
visual.set_title('Spring simulation', color= '#FFFFFF')
for spine in visual.spines.values():
    spine.set_edgecolor('#374151')



def init():
    obj.set_data([],[])
    spring_line.set_data([],[])
    wall.set_data([],[])
    return obj, spring_line, wall

def animate(i):
    obj.set_data([x[i]],[0])
    spring_line.set_data([-2,x[i]],[0])
    wall.set_data([-2],[-2,2])
    return obj, spring_line, wall

anim = animation.FuncAnimation( fig, animate, init_func=init,   frames=len(spring.time_steps), interval=spring.h * 1000, blit = False)

plt.tight_layout()
plt.show()