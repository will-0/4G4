import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML

fig, ax = plt.subplots()

ax.set_xlim(( 0, 1))
ax.set_ylim((0, 1))

line, = ax.plot([], [], marker="*", color="r", linestyle="None");

#add a background
x = np.linspace(0,1,10)
y = np.linspace(0,1,10)
z = np.zeros([10,10])
for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        z[i,j] = (x[i] - 0.5)**2 + (y[j] - 0.5)**2

x1,x2 = np.meshgrid(x,y);
background = ax.contour(x1, x2, -z, 8); #, cmap="gray")

fig, ax = plt.subplots()

ax.set_xlim(( 0, 1))
ax.set_ylim((0, 1))

line, = ax.plot([], [], marker="*", color="r", linestyle="None");

#add a background
x = np.linspace(0,1,10)
y = np.linspace(0,1,10)
z = np.zeros([10,10])
for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        z[i,j] = (x[i] - 0.5)**2 + (y[j] - 0.5)**2

x1,x2 = np.meshgrid(x,y);
background = ax.contour(x1, x2, -z, 8); #, cmap="gray")

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return (line,)

###Actual working area

from SwarmParty import Swarm

def sphere_func(position):
    if position.ndim == 1:
        position = position.reshape([1, position.shape[0]])
    return ((position[:,0]-0.5)**2 + (position[:,1]-0.5)**2).reshape([position.shape[0], 1])

my_swarm = Swarm(2, sphere_func, n_particles = 10, start_range = [0.0, 0.1], omega = 0.7, v_max = 0.1, c_1 = 2, c_2 = 2)

def animate(i):
    data = my_swarm.update_particles()
    line.set_data(data[:,0], data[:,1])
    return (line,)

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=100, interval=100, blit=True)
HTML(anim.to_html5_video())