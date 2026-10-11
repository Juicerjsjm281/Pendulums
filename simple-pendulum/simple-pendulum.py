import numpy as np
import matplotlib.pyplot as plt

def runge_kutta_4(f, t, Y , dt): # h es el salto de paso
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
h = 0.02

# Initial conditions
y0 = [np.pi / 2, 0]


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
    """It's the pendulum system equation with the assumption of sin(theta) ~= theta
    
    Keyword arguments:
        theta0:          Initial angular position
        omega0:          Initial angular velocity
    """
    max_amplitude = np.sqrt(theta0**2 + (omega0**2) /k_squared)
    initial_phase = np.arctan(- omega0 * theta0)
    return [max_amplitude, initial_phase]



####################

# Initial configuration
time_range = np.arange(0, 20+h, h)




sol_fsps = np.zeros((len(time_range), len(y0)))
sol_fsps[0] = y0

sol_asps = np.zeros((len(time_range), len(y0)))
sol_asps[0] = y0


# Resolution of fsps with rk4

for i in range(len(time_range)-1):
    sol_fsps[i+1] = runge_kutta_4(full_simple_pendulum_system, time_range[i], sol_fsps[i], h)


values = aproximation_simple_pendulum_system(y0[0], y0[1])
for i in range(len(time_range)-1):
    sol_asps[i+1] = [values[0] * np.cos(np.sqrt(k_squared) * time_range[i] + values[1] ), - np.sqrt(k_squared) * values[0] * np.sin(np.sqrt(k_squared) * time_range[i] + values[1])]

theta_fsps_sol = sol_fsps[:,0]
omega_fsps_sol = sol_fsps[:,1]

theta_asps_sol = sol_asps[:,0]
omega_asps_sol = sol_asps[:,1]




# Visual de la comparación entre fsps y asps

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (13,4.5))

ax1.plot(time_range, theta_fsps_sol, label='fsps',color = 'red')
ax1.plot(time_range, theta_asps_sol, label = ' asps' , color = 'blue')
ax1.grid()
ax1.legend(loc = 'upper right')

ax2.plot(theta_fsps_sol, omega_fsps_sol, label='fsps',color = 'red')
ax2.plot(theta_asps_sol, omega_asps_sol, label = ' asps' , color = 'blue')
ax2.legend(loc = 'upper right')
ax2.set_xlabel(r"$\theta \ [rad]$")
ax2.set_ylabel(r"$\omega \ [\frac{rad}{s}]$")

plt.show()