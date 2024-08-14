import osqp
import numpy as np
import scipy as sp
from scipy import sparse
from LQR_pin import InvertedPendulumLQR
import matplotlib.pyplot as plt

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

[nx, nu] = Bd.shape

# Constraints
u0 = 0
umin = np.array([-1.]) - u0
umax = np.array([1.]) - u0
xmin = np.array([-np.inf,-np.inf,-np.inf,-np.inf])
xmax = np.array([np.inf, np.inf, np.inf, np.inf])

# Objective function
Q = sparse.diags([0., 1.0, 100.0, 0.1])
QN = Q
R = 0.1*sparse.eye(1)

# Initial and reference states
x0 = np.array([0., 0., 0.2, 0.])
xr = np.array([0., 0., 0., 0.])

# Prediction horizon
N = 10

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
nsim = 500
x01_values = []
x02_values = []
ctrl_values = []

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

    # Store values for plotting
    x02_values.append(x0[2])
    x01_values.append(x0[1])
    ctrl_values.append(ctrl[0])


# Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# Plot x0
axs[0].plot(range(nsim), x01_values, label='x0')
axs[0].set_title('State x dot over time')
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('State')
axs[0].legend()
axs[0].grid(True)

# Plot x0
axs[1].plot(range(nsim), x02_values, label='x0')
axs[1].set_title('State theta over time')
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('State')
axs[1].legend()
axs[1].grid(True)


# Plot ctrl
axs[2].plot(range(nsim), ctrl_values, label='ctrl', color='orange')
axs[2].set_title('Control Input ctrl over time')
axs[2].set_xlabel('Iteration')
axs[2].set_ylabel('Control Input')
axs[2].legend()
axs[2].grid(True)


# Save the plot as an image file
plt.tight_layout()
plt.savefig('x0_ctrl_plot.png')
plt.show()