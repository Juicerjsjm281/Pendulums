import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class spring_mass:

    name = 'Mass-Spring'
    axes_names = [r"$x \ \[m]$", r"$y \ \[m]$"]

    def __init__(self,x1_0,x2_0,v1_0,v2_0,k,m_1,m_2, l_0,t_0 = 0,h=0.01): 
        self.y0 = np.array([x1_0 , x2_0, v1_0, v2_0])
        self.m_1, self.m_2, self.h, self.l_0, self.k =m_1, m_2 , h, l_0, k
        self.time_span = (t_0, t_0 + 5)
        self.t = np.arange(t_0 , t_0 +5, h)
        self.solution = None

    def fode(self,t,y):

        x1 , x2, v1, v2 = y

        x21 = (x2-x1) - self.l_0
        
        dx1 = v1
        dx2 = v2
        dv1 = (self.k / self.m_1) * x21
        dv2 = - (self.k / self.m_2) * x21
        return np.array([dx1,dx2,dv1,dv2])


    def solution_method(self):
        sol = solve_ivp(fun = self.fode,t_span = self.time_span, y0 = self.y0, t_eval= self.t)
        self.solution = sol.y.T
        return self.solution

    def spring_graph(self, xs_start, xs_end, n_coils = 6, amp = 0.3, connection = 0.1):
        distance = xs_end - xs_start
        c = np.sign(distance) * min(connection, 0.2 * abs(distance))

        n_pts = 4*n_coils + 1
        xs_line = np.linspace(xs_start+c,xs_end-c, n_pts)
        ys_line = np.zeros(n_pts)
        ys_line[1::4]=amp
        ys_line[3::4]=-amp
        xs_graph = np.concatenate([[xs_start],xs_line,[xs_end]])
        ys_graph = np.concatenate([[0],ys_line,[0]])
        return xs_graph, ys_graph



#The spring model with its solution

spring = spring_mass(1, 2, 0 , 2, 3, 1,1,3)

spring_solution = spring.solution_method()

x1 = spring_solution[:,0]
x2 = spring_solution[:,1]
v1 = spring_solution[:,2]
v2 = spring_solution[:,3]

ke = 0.5 * (spring.m_1 * (v1**2) + spring.m_2 * (v2**2))
pe = 0.5 * spring.k * ((x2 - x1 - spring.l_0)**2)
te = ke + pe





#Graphing

fig, axs = plt.subplots(1,2,figsize = (12,5), facecolor = "#0B0F19")

visual = axs[0]
energy = axs[1]

obj1, = visual.plot([],[],'o', label='Mass 1', color = '#E60033', zorder = 3)
obj2, = visual.plot([],[],'o', label = 'Mass 2',color = '#39C5BB',zorder = 2)
spring_line, = visual.plot([],[],'--',label='Spring',color = '#FFD700',zorder = 2)
visual.set_xlim(np.min([x1,x2]) -0.2,np.max([x1,x2]) + 0.2)
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



ke_graph, = energy.plot([],[],'--', label='Kinetic Energy', color = '#E0A96D', zorder = 2)
pe_graph, = energy.plot([],[],'--',label='Potential Energy',color = '#C9CDD6',zorder = 2)
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
    for element in (obj1, obj2,spring_line,ke_graph,pe_graph,te_graph):
        element.set_data([],[])
    return obj1, obj2, spring_line, ke_graph, pe_graph, te_graph

def animate(i):
    obj1.set_data([x1[i]],[0])
    obj2.set_data([x2[i]],[0])

    
    xs, ys = spring.spring_graph(x1[i],x2[i])
    spring_line.set_data(xs,ys)

    ke_graph.set_data(spring.t[:i+1],ke[:i+1])
    pe_graph.set_data(spring.t[:i+1],pe[:i+1])
    te_graph.set_data(spring.t[:i+1],te[:i+1])
    return obj1, obj2, spring_line, ke_graph, pe_graph, te_graph

anim = animation.FuncAnimation( fig, animate, init_func=init,   frames=len(spring.t), interval=spring.h * 1000, blit = False)

plt.tight_layout()
plt.show()