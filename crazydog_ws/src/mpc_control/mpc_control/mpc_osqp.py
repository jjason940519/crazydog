import osqp
import numpy as np
import scipy as sp
from scipy import sparse
from LQR_pin import InvertedPendulumLQR
import matplotlib.pyplot as plt
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
                                    delta_t=1/300, 
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
umin = np.array([-1.]) - u0
umax = np.array([1.]) - u0
xmin = np.array([-np.inf,-np.inf,-np.inf,-np.inf])
xmax = np.array([np.inf, np.inf, np.inf, np.inf])

# Objective function
Q = sparse.diags([0., 1.0, 100.0, 0.005])
QN = Q
R = 0.001*sparse.eye(1)

# Initial and reference states
x0 = np.array([0., 0., 0.2, 0.])
xr = np.array([0., 0., 0., 0.])

# Prediction horizon
N = 50

# Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
# - quadratic objective
P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                       sparse.kron(sparse.eye(N), R)], format='csc')
# - linear objective
q = np.hstack([np.kron(np.ones(N), -Q@xr), -QN@xr, np.zeros(N*nu)])
# - linear dynamics
Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
Aeq = sparse.hstack([Ax, Bu])
leq = np.hstack([-x0, np.zeros(N*nx)])
ueq = leq
# - input and state constraints
Aineq = sparse.eye((N+1)*nx + N*nu)
lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
# - OSQP constraints
A = sparse.vstack([Aeq, Aineq], format='csc')
l = np.hstack([leq, lineq])
u = np.hstack([ueq, uineq])

# Create an OSQP object
prob = osqp.OSQP()

# Setup workspace
prob.setup(P, q, A, l, u)

# Simulate in closed loop
nsim = 1000
x01_values = []
x02_values = []
x03_values = []
x00_values = []
ctrl_values = []

t0 = time.time()
for i in range(nsim):
    # Solve
    res = prob.solve()

    # Check solver status
    if res.info.status != 'solved':
        raise ValueError('OSQP did not solve the problem!')

    # Apply first control input to the plant
    ctrl = res.x[-N*nu:-(N-1)*nu]
    x0 = Ad@x0 + Bd@ctrl

    # Update initial state
    l[:nx] = -x0
    u[:nx] = -x0
    prob.update(l=l, u=u)
    print('i', i)
    print('X', x0)
    print('u', ctrl)

    t1 = time.time()
    print('freq', 1/(t1-t0))
    t0 = t1


    # Store values for plotting
    x02_values.append(x0[2])
    x01_values.append(x0[1])
    x00_values.append(x0[0])
    x03_values.append(x0[3])
    ctrl_values.append(ctrl[0])


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
plt.savefig('osqp.png')
plt.show()