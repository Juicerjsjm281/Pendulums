import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def runge_kutta_4(f, t, Y , dt): # h is the step taken
    """To Use The Fourth-Order Runge-Kutta Method.
    
    Keyword arguments:
        f (callable):               The function that describes the system.
        t (float):                  Time
        Y ([theta,omega]):          The values in the time[i]
        h (float):                  Step      
    """ 
    k_1 = f(t, Y)
    k_2 = f(t + 0.5 * dt, Y + 0.5 * dt * k_1)
    k_3 = f(t + 0.5 * dt, Y + 0.5 * dt * k_2)
    k_4 = f(t + dt, Y + dt * k_3)
    return Y + (dt/6.0) * (k_1 + 2*(k_2+k_3) + k_4)     # Integration of the i + 1 value for Y


h = 0.01

# Parameters
g = 9.81 # m/s^2 Gravity value
l_1 = 1 # m 
l_2 = 1
m_1 = 1  #kg
m_2 = 1 #kg

# Initial conditions
y0 = np.array([np.pi , np.pi / 3, 0, 0])  # theta_1(t=0), theta_2(t=0), omega_1(t=0), omega_2(t=0)


# Constants pre-calculated to avoid re-calculating them many times
m_2_12 = m_2 / (m_1+m_2)
l_1_2 = l_1 / l_2
l_2_1 = l_2 / l_1
gl_1 = g / l_1
gl_2 = g / l_2

def double_pendulum_system(t, Y):    #fsps
    """It's the pendulumn system equation for a double pendulum
    
    Keyword arguments:
        Y[0]:        theta_1
        Y[1]:        theta_2
        Y[2]:        omega_1
        y[3]:        omega_2
    """
    theta_1, theta_2, omega_1, omega_2  = Y                       # Current values for the system

    delta = theta_1 - theta_2

    f_1 = - m_2_12 * l_2_1 * np.sin(delta) * (omega_2**2) - gl_1 * np.sin(theta_1)
    f_2 = l_1_2 * np.sin(delta) * (omega_1**2) - gl_2 * np.sin(theta_2)

    alpha_1 = m_2_12 * l_2_1 * np.cos(delta)
    alpha_2 = l_1_2 * np.cos(delta)

    det = 1 - alpha_1 * alpha_2

    dtheta_1_dt = omega_1                       # Angular velocity for the first pendulum
    dtheta_2_dt = omega_2                       # Angular velocity for the second pendulum
    domega_1_dt = (f_1 - alpha_1 * f_2) / det
    domega_2_dt = (f_2 - alpha_2 * f_1) /  det
    return np.array([dtheta_1_dt, dtheta_2_dt, domega_1_dt, domega_2_dt])

# Time Definition
time_range = np.arange(0, 10 + h, h)

sol_dps = np.zeros((len(time_range), len(y0)))
sol_dps[0] = y0


for i in range(len(time_range) - 1):
    sol_dps[i+1] = runge_kutta_4(double_pendulum_system, time_range[i], sol_dps[i], h)


theta_1_dps_sol = sol_dps[:,0]
theta_2_dps_sol = sol_dps[:,1]
omega_1_dps_sol = sol_dps[:,2]
omega_2_dps_sol = sol_dps[:,3]


x1 =  l_1 * np.sin(theta_1_dps_sol)
y1 = -l_1 * np.cos(theta_1_dps_sol)
x2 = x1 + l_2 * np.sin(theta_2_dps_sol)
y2 = y1 - l_2 * np.cos(theta_2_dps_sol)

ke_1 = 0.5 * m_1 * (l_1**2) * (omega_1_dps_sol**2)
pe_1 = - m_1 * g * np.cos(theta_1_dps_sol)
te_1 = ke_1 + pe_1

ke_2 = 0.5* m_2 * ((l_1**2) * (omega_1_dps_sol**2) + (l_2**2) * (omega_2_dps_sol**2) + 2*l_1*l_2*omega_1_dps_sol*omega_2_dps_sol*np.cos(theta_1_dps_sol-theta_2_dps_sol))
pe_2 = - m_2* g* (l_1 * np.cos(theta_1_dps_sol) + l_2*np.cos(theta_2_dps_sol))
te_2 = ke_2 + pe_2

te_t = ke_1+ke_2+pe_1+pe_2

double_pendulum_figure, axs = plt.subplots(2,2, figsize = (12,9), facecolor= '#0B0F19')

visual = axs[0,0]
energy_t = axs[0,1]
energy_1 = axs[1,0]
energy_2 = axs[1,1]


visual_pendulum_1, = visual.plot([],[],'o-', label='Pendulum 1', color = '#E60033', zorder = 3)
visual_pendulum_2, = visual.plot([],[],'o-', label='Pendulum 2',color = '#39C5BB',zorder = 2)
visual.set_xlim(np.min(x2)-0.2,np.max(x2) + 0.2)
visual.set_ylim(np.min(y2)-0.2, np.max(y2)+0.2)
visual.set_xlabel(r"$x \ [m]$")
visual.set_ylabel(r"$y \ [m]$")
visual.xaxis.label.set_color('#E5E7EB')
visual.yaxis.label.set_color('#E5E7EB')
visual.set_aspect('equal', adjustable='box')
visual.tick_params(colors='#9CA3AF', labelsize=10)
visual.set_facecolor('#111827')
legend_visual = visual.legend(loc = 'upper right', frameon = False)
for text in legend_visual.get_texts():
    text.set_color('#FFFFFF')
visual.grid(color = '#374151')
visual.set_title('Double-Pendulum simulation', color= '#FFFFFF')
for spine in visual.spines.values():
    spine.set_edgecolor('#374151')


te_m1, = energy_t.plot([],[],'--', label='Total Energy 1',color = '#E60033', alpha = 0.5)
te_m2, = energy_t.plot([],[],'--', label='Total Energy 2', color = '#39C5BB', alpha=0.5)
te_pendulum, = energy_t.plot([],[],'-', label='Total Energy', color = '#FFD700')
energy_t.set_xlim(np.min(time_range) -0.2 , np.max(time_range) + 0.2 )
energy_t.set_ylim(np.min([te_1,te_2,te_t])-0.5, np.max([te_1,te_2,te_t])+0.5)
energy_t.set_xlabel(r"$t \ [s]$")
energy_t.set_ylabel(r"$E \ [J]$")
energy_t.xaxis.label.set_color('#E5E7EB')
energy_t.yaxis.label.set_color('#E5E7EB')
energy_t.tick_params(colors='#9CA3AF', labelsize=10)
energy_t.set_facecolor('#111827')
legend_energy_t= energy_t.legend(loc = 'upper right', frameon = False)
for text in legend_energy_t.get_texts():
    text.set_color('#FFFFFF')
energy_t.grid(color = '#374151')
energy_t.set_title('Double Pendulum Energy', color= '#FFFFFF')
for spine in energy_t.spines.values():
    spine.set_edgecolor('#374151')


ke_pendulum_1, = energy_1.plot([],[],'--', label='Kinetic Energy',color = '#E0A96D', alpha = 0.5)
pe_pendulum_1, = energy_1.plot([],[],'--', label='Potential Energy', color = '#C9CDD6', alpha=0.5)
te_pendulum_1, = energy_1.plot([],[],'-', label='Total Energy', color = '#E60033')
energy_1.set_xlim(np.min(time_range) -0.2 , np.max(time_range) + 0.2 )
energy_1.set_ylim(np.min([ke_1,pe_1,te_1])-0.5, np.max([ke_1,pe_1,te_1])+0.5)
energy_1.set_xlabel(r"$t \ [s]$")
energy_1.set_ylabel(r"$E \ [J]$")
energy_1.xaxis.label.set_color('#E5E7EB')
energy_1.yaxis.label.set_color('#E5E7EB')
energy_1.tick_params(colors='#9CA3AF', labelsize=10)
energy_1.set_facecolor('#111827')
legend_energy_1= energy_1.legend(loc = 'upper right', frameon = False)
for text in legend_energy_1.get_texts():
    text.set_color('#FFFFFF')
energy_1.grid(color = '#374151')
energy_1.set_title('Pendulum 1 Energy', color= '#FFFFFF')
for spine in energy_1.spines.values():
    spine.set_edgecolor('#374151')


ke_pendulum_2, = energy_2.plot([],[],'--', label='Kinetic Energy',color = '#E12885', alpha= 0.5)
pe_pendulum_2, = energy_2.plot([],[],'--', label='Potential Energy', color = '#B4B9C2', alpha = 0.5)
te_pendulum_2, = energy_2.plot([],[],'-', label='Total Energy', color = '#39C5BB')
energy_2.set_xlim(np.min(time_range) -0.2 , np.max(time_range) + 0.2 )
energy_2.set_ylim(np.min([ke_2,pe_2,te_2])-0.5, np.max([ke_2,pe_2,te_2])+0.5)
energy_2.set_xlabel(r"$t \ [s]$")
energy_2.set_ylabel(r"$E \ [J]$")
energy_2.xaxis.label.set_color('#E5E7EB')
energy_2.yaxis.label.set_color('#E5E7EB')
energy_2.tick_params(colors='#9CA3AF', labelsize=10)
energy_2.set_facecolor('#111827')
legend_energy_2= energy_2.legend(loc = 'upper right', frameon = False)
for text in legend_energy_2.get_texts():
    text.set_color('#FFFFFF')
energy_2.grid(color = '#374151')
energy_2.set_title('Pendulum 2 Energy', color= '#FFFFFF')
for spine in energy_2.spines.values():
    spine.set_edgecolor('#374151')

def init():
    visual_pendulum_1.set_data([],[])
    visual_pendulum_2.set_data([],[])

    ke_pendulum_1.set_data([],[])
    pe_pendulum_1.set_data([],[])
    te_pendulum_1.set_data([],[])

    ke_pendulum_2.set_data([],[])
    pe_pendulum_2.set_data([],[])
    te_pendulum_2.set_data([],[])

    te_m1.set_data([],[])
    te_m2.set_data([],[])
    te_pendulum.set_data([],[])
    return visual_pendulum_1, visual_pendulum_2, ke_pendulum_1, pe_pendulum_1, te_pendulum_1, ke_pendulum_2, pe_pendulum_2, te_pendulum_2, te_m1, te_m2, te_pendulum

def animate(i):
    visual_pendulum_1.set_data([0, x1[i]],[0, y1[i]])
    visual_pendulum_2.set_data([x1[i],x2[i]],[y1[i],y2[i]])

    ke_pendulum_1.set_data(time_range[:i+1],ke_1[:i+1])
    pe_pendulum_1.set_data(time_range[:i+1], pe_1[:i+1])
    te_pendulum_1.set_data(time_range[:i+1], te_1[:i+1])


    ke_pendulum_2.set_data(time_range[:i+1],ke_2[:i+1])
    pe_pendulum_2.set_data(time_range[:i+1], pe_2[:i+1])
    te_pendulum_2.set_data(time_range[:i+1], te_2[:i+1])

    te_m1.set_data(time_range[:i+1], te_1[:i+1])
    te_m2.set_data(time_range[:i+1], te_2[:i+1])
    te_pendulum.set_data(time_range[:i+1], te_t[:i+1])
    return visual_pendulum_1, visual_pendulum_2, ke_pendulum_1, pe_pendulum_1, te_pendulum_1, ke_pendulum_2, pe_pendulum_2, te_pendulum_2, te_pendulum, te_m1, te_m2

ani = animation.FuncAnimation(fig=double_pendulum_figure, func= animate, init_func=init, frames = range(0, len(time_range), 5), interval = h * 5 * 1000, blit = False)


plt.tight_layout()
plt.show()