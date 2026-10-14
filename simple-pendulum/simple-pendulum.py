import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def runge_kutta_4(f, t, Y , dt): # h is the step taken
    """To Use The Fourth-Order Runge-Kutta Method.
    
    Keyword arguments:
        f (callable):               The function that describes the system.
        t (float):                  Time
        Y ([theta,omega]):          Solutions
        h (float):                  Step      
    """ 
    k_1 = f(t, Y)
    k_2 = f(t + 0.5 * dt, Y + 0.5 * dt * k_1)
    k_3 = f(t + 0.5 * dt, Y + 0.5 * dt * k_2)
    k_4 = f(t + dt, Y + dt * k_3)
    return Y + (dt/6.0) * (k_1 + 2*(k_2+k_3) + k_4)     # Integration of the i + 1 value for Y

####################

# Parameters
g = 9.81 # m/s^2 Gravity value
l = 1 # m 
k_squared = g/l # 1/s^2 Angular frecuency
h = 0.01

# Initial conditions
y0 = [np.pi / 4, 0]


###################

def full_simple_pendulum_system(t, Y):    #fsps
    """It's the pendulumn system equation derived from the substitution of omega = dtheta_dt
    
    Keyword arguments:
        Y[0]:        theta
        Y[1]:        omega
    """
    theta, omega  = Y                       # Current values for the system

    dtheta_dt = omega                       # Angular velocity
    domega_dt = - k_squared * np.sin(theta) # Angular acceleration
    return np.array([dtheta_dt,domega_dt])


def aproximation_simple_pendulum_system(theta0, omega0):
    """It's the values for the maximum amplitude and initial phase for the pendulum system equation with the assumption of sin(theta) ~= theta
    
    Keyword arguments:
        theta0:          Initial angular position
        omega0:          Initial angular velocity
    """
    max_amplitude = np.sqrt(theta0**2 + (omega0**2 /k_squared)) 
    initial_phase = np.arctan(- omega0 / (np.sqrt(k_squared)* theta0))
    return max_amplitude, initial_phase



####################

# Initial configuration
time_range = np.arange(0,8+ h, h)


sol_fsps = np.zeros((len(time_range), len(y0)))
sol_fsps[0] = y0


# Resolution of fsps with rk4

for i in range(len(time_range)-1):
    sol_fsps[i+1] = runge_kutta_4(full_simple_pendulum_system, time_range[i], sol_fsps[i], h)


a, phi = aproximation_simple_pendulum_system(y0[0], y0[1])


theta_fsps_sol = sol_fsps[:,0] 
omega_fsps_sol = sol_fsps[:,1]

theta_asps_sol = a * np.cos(np.sqrt(k_squared) * time_range + phi)
omega_asps_sol = - a * np.sqrt(k_squared) * np.sin(np.sqrt(k_squared) * time_range + phi)




# Visual de la comparación entre fsps y asps
simple_pendulum_figure, axs = plt.subplots(2,2, figsize = (12,9), facecolor= '#0B0F19')

#visual = axs[0,0]
#energy = axs[0,1]
ax1 = axs[1,0]
ax2 = axs[1,1]

simple_pendulum_figure.delaxes(axs[0, 0])
simple_pendulum_figure.delaxes(axs[0, 1])

visual = simple_pendulum_figure.add_subplot(2, 1, 1)

visual_fsps, = visual.plot([], [],'o-', label='fsps', color = '#E60033')
visual_asps, = visual.plot([],[],'o-', label='asps',color = '#39C5BB')
visual.set_xlim(-l-0.5,l+0.5)
visual.set_ylim(-l-0.5, l+ 0.5)
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
visual.set_title('Pendulum simulation', color= '#FFFFFF')
for spine in visual.spines.values():
    spine.set_edgecolor('#374151')





#First subplot (time vs theta)
ax1_fsps, = ax1.plot([], [], label='fsps',color = '#E60033', alpha=0.7)
ax1_asps, = ax1.plot([], [], label = ' asps' , color = '#39C5BB', alpha = 0.7)
ax1.set_xlim(time_range[0],time_range[-1])
ax1.set_ylim(np.min(theta_fsps_sol) - 0.5, np.max(theta_fsps_sol) + 0.5)
ax1.set_xlabel(r"$t \ [s]$")
ax1.set_ylabel(r"$\theta \ [rad]$")
ax1.xaxis.label.set_color('#E5E7EB')
ax1.yaxis.label.set_color('#E5E7EB')
ax1.tick_params(colors='#9CA3AF', labelsize=10)
ax1.grid(color = '#374151')
ax1.set_facecolor('#111827')
legend1 = ax1.legend(loc = 'upper right', frameon = False)
for text in legend1.get_texts():
    text.set_color('#FFFFFF')
ax1.set_title(r"$\theta (t)$", color= '#FFFFFF')
for spine in ax1.spines.values():
    spine.set_edgecolor('#374151')

#Second subplot (phase diagram)
ax2_fsps, = ax2.plot([], [], label='fsps',color = '#E60033')
ax2_asps, = ax2.plot([], [], label = ' asps' , color = '#39C5BB')
ax2.set_xlim(np.min(theta_fsps_sol) -0.5, np.max(theta_fsps_sol) + 0.5)
ax2.set_ylim(np.min(omega_asps_sol) -0.5, np.max(omega_asps_sol) + 0.5)
ax2.set_xlabel(r"$\theta \ [rad]$")
ax2.set_ylabel(r"$\omega \ [\frac{rad}{s}]$")
ax2.xaxis.label.set_color('#E5E7EB')
ax2.yaxis.label.set_color('#E5E7EB')
ax2.tick_params(colors='#9CA3AF', labelsize=10)
ax2.grid(color = '#374151')
ax2.set_facecolor('#111827')
legend2 = ax2.legend(loc = 'upper right', frameon = False)
for text in legend2.get_texts():
    text.set_color('#FFFFFF')
ax2.set_title(r"Diagram phase of $\theta $ and $ \omega$", color= '#FFFFFF')
for spine in ax2.spines.values():
    spine.set_edgecolor('#374151')


#Section to define how the animation is gonna be

def init():
    """init function for animation.FuncAnimation()"""
    ax1_fsps.set_data([],[])
    ax1_asps.set_data([],[])

    ax2_fsps.set_data([],[])
    ax2_asps.set_data([],[])

    visual_fsps.set_data([],[])
    visual_asps.set_data([],[])
    return ax1_fsps, ax1_asps, ax2_fsps, ax2_asps, visual_fsps, visual_asps


def animate(i):
    """Values for the trace in each frame of the animation"""
    ax1_fsps.set_data(time_range[:i+1], theta_fsps_sol[:i+1])
    ax1_asps.set_data(time_range[:i+1], theta_asps_sol[:i+1])


    ax2_fsps.set_data(theta_fsps_sol[:i+1], omega_fsps_sol[:i+1])
    ax2_asps.set_data(theta_asps_sol[:i+1], omega_asps_sol[:i+1])


    visual_fsps.set_data([0,l * np.sin(theta_fsps_sol[i])], [0,-l * np.cos(theta_fsps_sol[i])])
    visual_asps.set_data([0,l * np.sin(theta_asps_sol[i])], [0,-l * np.cos(theta_asps_sol[i])])
    return ax1_fsps, ax1_asps, ax2_fsps, ax2_asps, visual_fsps, visual_asps


ani = animation.FuncAnimation(fig=simple_pendulum_figure, func= animate, init_func=init, frames = len(time_range), interval = h * 1000, blit = True)

# ani.save('pendulum_animation.gif', writer='pillow', fps=int(len(time_range)/(3+h)))

plt.tight_layout()
plt.show()