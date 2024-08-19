import numpy as np
import scipy as sp
from scipy import sparse
from LQR_pin import InvertedPendulumLQR
import matplotlib.pyplot as plt

from cvxpy import *
import numpy as np
import scipy as sp
from scipy import sparse
import time

WHEEL_RADIUS = 0.08     # m
WHEEL_MASS = 0.695  # kg
URDF_PATH = "/home/crazydog/crazydog/crazydog_ws/src/mpc_control/mpc_control/robot_models/big bipedal robot v1/urdf/big bipedal robot v1.urdf"

pos = np.array([0., 0., 0., 0., 0., 0., 1.,
                0., -1.18, 2.0, 1., 0.,
                0., -1.18, 2.0, 1., 0.])
lqr_controller = InvertedPendulumLQR(pos=pos, 
                                    urdf=URDF_PATH, 
                                    wheel_r=WHEEL_RADIUS, 
                                    M=WHEEL_MASS, 
                                    delta_t=1/100, 
                                    show_animation=False)
Ad = sparse.csc_matrix(lqr_controller.A.tolist())
Bd = sparse.csc_matrix(lqr_controller.B.tolist())
# l_bar = 2.0  # length of bar
# M = 1.0  # [kg]
# m = 0.3  # [kg]
# g = 9.8  # [m/s^2]
# delta_t = 1/100
# nx = 4  # number of state
# nu = 1  # number of input
# A = np.array([
#         [0.0, 1.0, 0.0, 0.0],
#         [0.0, 0.0, m * g / M, 0.0],
#         [0.0, 0.0, 0.0, 1.0],
#         [0.0, 0.0, g * (M + m) / (l_bar * M), 0.0]
#     ])
# A = np.eye(nx) + delta_t * A

# B = np.array([
#     [0.0],
#     [1.0 / M],
#     [0.0],
#     [1.0 / (l_bar * M)]
# ])
# B = delta_t * B
# Ad = sparse.csc_matrix(lqr_controller.A.tolist())
# Bd = sparse.csc_matrix(lqr_controller.B.tolist())


[nx, nu] = Bd.shape

# Constraints
u0 = 0
umin = np.array([-5.]) - u0
umax = np.array([5.]) - u0
xmin = np.array([-np.inf,-np.inf,-np.inf,-np.inf])
xmax = np.array([np.inf, np.inf, np.inf, np.inf])

# Objective function
Q = sparse.diags([0.0, 1.0, 1000.0, 0.1])
QN = Q
R = 0.1*sparse.eye(1)

# Initial and reference states
x0 = np.array([0., 0., 0.2, 0.])
xr = np.array([0., 0., 0., 0.])

# Prediction horizon
N = 10

# Define problem
u = Variable((nu, N))
x = Variable((nx, N+1))
x_init = Parameter(nx)
objective = 0
constraints = [x[:,0] == x_init]
for k in range(N):
    objective += quad_form(x[:,k] - xr, Q) + quad_form(u[:,k], R)
    constraints += [x[:,k+1] == Ad@x[:,k] + Bd@u[:,k]]
    constraints += [xmin <= x[:,k], x[:,k] <= xmax]
    constraints += [umin <= u[:,k], u[:,k] <= umax]
objective += quad_form(x[:,N] - xr, QN)
prob = Problem(Minimize(objective), constraints)

# Simulate in closed loop
nsim = 500
x01_values = []
x02_values = []
x03_values = []
x00_values = []
ctrl_values = []

t0 = time.time()
for i in range(nsim):
    x_init.value = x0
    prob.solve(solver=OSQP, warm_start=True)
    x0 = Ad@x0 + Bd@u[:,0].value

    print('i', i)
    print('X', x0)
    print('u', u[:,0].value)

    # Store values for plotting
    x02_values.append(x0[2])
    x01_values.append(x0[1])
    x00_values.append(x0[0])
    x03_values.append(x0[3])
    ctrl_values.append(u[:,0].value)

    t1 = time.time()
    print('freq', 1/(t1-t0))
    t0 = t1


# Plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot x0
axs[0].plot(range(nsim), x00_values, label='x')
axs[0].plot(range(nsim), x01_values, label='x_dot')
axs[0].plot(range(nsim), x02_values, label='theta')
axs[0].plot(range(nsim), x03_values, label='theta_dot')
axs[0].set_title('State x dot over time')
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('State')
axs[0].legend()
axs[0].grid(True)


# Plot ctrl
axs[1].plot(range(nsim), ctrl_values, label='ctrl', color='orange')
axs[1].set_title('Control Input ctrl over time')
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Control Input')
axs[1].legend()
axs[1].grid(True)


# Save the plot as an image file
plt.tight_layout()
plt.savefig('cvxpy.png')
plt.show()