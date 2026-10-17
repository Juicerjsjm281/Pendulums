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
        self.time_span = (t_0, t_0 + 5)
        self.t = np.arange(t_0 , t_0 +5, h)
        self.solution = None

    def fode(self,t,y):

        x , v = y
        
        dx = v
        dv = - (self.k / self.m) * x
        return np.array([dx,dv])


    def solution_method(self):    #It would probably be better to use the analitic solution but the objective here was to learn to use scipy.integrate.solve_ivp()
        sol = solve_ivp(fun = self.fode,t_span = self.time_span,y0 = self.y0, t_eval= self.t)
        self.solution = sol.y.T
        return self.solution


spring = spring_mass(1, 0, 2, 1)

spring_solution = spring.solution_method()

x = spring_solution[:,0]
v = spring_solution[:,1]
ke = 0.5 * spring.m * (v**2)
pe = 0.5 * spring.k * (x**2)
te = ke + pe



def spring_graph(xs_start, xs_end, n_coils = 6, amp = 0.3, conection = 0.1):
    n_pts = 4*n_coils + 1
    xs_line = np.linspace(xs_start+conection,xs_end-conection, n_pts)
    ys_line = np.zeros(n_pts)
    ys_line[1::4]=amp
    ys_line[3::4]=-amp
    xs_graph = np.concatenate([[xs_start],xs_line,[xs_end]])
    ys_graph = np.concatenate([[0],ys_line,[0]])
    return xs_graph, ys_graph

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
visual.set_title('Spring-Mass Simulation', color= '#FFFFFF')
for spine in visual.spines.values():
    spine.set_edgecolor('#374151')



ke_graph, = energy.plot([],[],'-', label='Kinetic Energy', color = '#E0A96D', zorder = 2)
pe_graph, = energy.plot([],[],'-',label='Potential Energy',color = '#C9CDD6',zorder = 2)
te_graph, = energy.plot([],[],'-', label = 'Total Energy',color = '#E60033',zorder = 3)
energy.set_xlim(-0.5, spring.t[-1] + 0.5)
energy.set_ylim(np.min([ke,pe,te]) -0.5, np.max([ke,pe,te])+0.5)
energy.xaxis.label.set_color('#E5E7EB')
energy.yaxis.label.set_color('#E5E7EB')
energy.tick_params(colors='#9CA3AF', labelsize=10)
energy.set_facecolor('#111827')
legend_energy = energy.legend(loc = 'upper right', frameon = False)
for text in legend_energy.get_texts():
    text.set_color('#FFFFFF')
energy.grid(color = '#374151')
energy.set_title('Spring-Mass Energy', color= '#FFFFFF')
for spine in energy.spines.values():
    spine.set_edgecolor('#374151')




def init():
    for element in (obj,spring_line,wall,ke_graph,pe_graph,te_graph):
        element.set_data([],[])
    return obj, spring_line, wall, ke_graph, pe_graph, te_graph

def animate(i):
    obj.set_data([x[i]],[0])

    
    wall.set_data([-2,-2],[-2,2])
    xs, ys = spring_graph(-2,x[i])
    spring_line.set_data(xs,ys)

    ke_graph.set_data(spring.t[:i+1],ke[:i+1])
    pe_graph.set_data(spring.t[:i+1],pe[:i+1])
    te_graph.set_data(spring.t[:i+1],te[:i+1])
    return obj, spring_line, wall, ke_graph, pe_graph, te_graph

anim = animation.FuncAnimation( fig, animate, init_func=init,   frames=len(spring.t), interval=spring.h * 1000, blit = True)

plt.tight_layout()
plt.show()