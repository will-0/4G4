import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

class Swarm():
    
    def __init__(self, n_dims, function, n_particles = 30, start_range = [-1, 1], v_max = 0.1, c_1 = 2, c_2 = 2, omega = 1, deterministic = False):
        #setup all the parameters of the swarm

        #positional (essential) arguments
        self.n_particles = n_particles
        self.n_dims = n_dims
        self.function = function

        #keyword (optional) arguments
        self.x_min = start_range[0]
        self.x_max = start_range[1]
        self.v_min = -1*v_max
        self.v_max = v_max
        self.c_1 = c_1
        self.c_2 = c_2
        self.omega = omega

        #set the initial positions
        self.current_pos = self.x_min + (self.x_max-self.x_min)*np.random.rand(self.n_particles, self.n_dims) #initialise the particles
        self.p_best = self.current_pos
        self.v = self.v_min + (self.v_max-self.v_min)*np.random.rand(self.n_particles, self.n_dims)

        self.g_best = self.p_best[np.argmin(self.f(self.current_pos)),:]
        self.pbest_perform = self.f(self.p_best)
        self.gbest_perform = self.f(self.g_best)
        
        #Evaluate the current positions
        self.g_best = self.p_best[np.argmin(self.f(self.current_pos)),:]
        self.pbest_perform = self.f(self.p_best)
        self.gbest_perform = self.f(self.g_best)

        #(ignore this vaiable: it's just a method to compare current pos to pbest/gbest etc.)
        self.is_better = np.zeros([self.n_particles,1])

        self.deterministic = deterministic
    
    def f(self, positions):
        return self.function(positions)

    def update_particles(self):
        """ Function to update particle positions using the PSO algorithm"""

        #exponential velocity decay
        # m_v_max = self.v_max#*np.exp(epoch/num_epochs)

        #update the current velocity
        self.v *= self.omega
        if not self.deterministic:
            self.v += self.c_1*(np.random.rand(self.current_pos.shape[0],1))*(self.p_best-self.current_pos)
            self.v += self.c_2*(np.random.rand(self.current_pos.shape[0],1))*(self.g_best-self.current_pos)
        else:
            self.v += self.c_1*0.5*(self.p_best-self.current_pos)
            self.v += self.c_2*0.5*(self.g_best-self.current_pos)
        # v_norm = np.linalg.norm(self.v,axis=1).reshape([self.v.shape[0],1])             #code for velocity limitation
        # self.v = np.where(v_norm < m_v_max, self.v, m_v_max*self.v/v_norm)                       #(comment in to use it)

        #update the current position
        self.current_pos += self.v

        #evaluate all of the particle's positions
        curr_perform = self.f(self.current_pos)
        
        #replace the p_bests with the current location if they are better
        self.is_better = (curr_perform<self.pbest_perform).reshape([self.is_better.shape[0],1])
        self.p_best = self.is_better*self.current_pos + np.logical_not(self.is_better)*self.p_best
        self.pbest_perform = self.is_better*curr_perform + np.logical_not(self.is_better)*self.pbest_perform
        
        #update g_best
        if np.min(self.pbest_perform) < self.gbest_perform:
            self.g_best = self.p_best[np.argmin(self.pbest_perform),:]
            self.gbest_perform = np.min(self.pbest_perform)

        return self.current_pos

class ObjectiveFunction():
    def __init__(self, func, minimum):
        self.minimum = minimum
        self.func = func

def investigate(a, b, objfunc, niter = 500, deterministic = False, clip = None):
    '''Takes in a (omega), b ([c1 + c2]/2) and an objective function, plots the particle trajectories for dimension 1 of an optimization of a 2D function'''
    my_swarm = Swarm(2, objfunc.func, n_particles = 10, start_range = [-5, 5], omega = a, v_max = 0.1, c_1 = b, c_2 = b, deterministic = deterministic)

    gbest = []
    trajectories = np.zeros([niter,10])
    for i in range(niter):
        my_swarm.update_particles()
        if clip == None:
            trajectories[i,:] = my_swarm.current_pos[:,0]
        else:
            trajectories[i,:] = np.where(np.abs(my_swarm.current_pos[:,0])<clip,my_swarm.current_pos[:,0],clip*np.sign(my_swarm.current_pos[:,0]))
        gbest.append(my_swarm.g_best[0])
    
    plt.plot(trajectories);
    plt.plot(objfunc.minimum[0]*(trajectories < 1000000), 'k');
    plt.xlabel('Number of iterations')
    plt.ylabel('Position')

class InvestigationObject():
    def __init__(self):
        pass

def get_error(swarm, objfunc):
    return (swarm.g_best-objfunc.minimum)@(swarm.g_best-objfunc.minimum)

def numerical_investigate(a, b, objfunc, n_to_average = 50, n_swarm_iters = 101):
    '''Takes in a combination of a and b, averages stats over 50 runs, and returns a return object with certain parameters'''
    returnObj = InvestigationObject()
    
    #define variables to extract
    gbest20 = []
    gbest50 = []
    gbest100 = []
    gbest500 = []

    var20 = []
    var50 = []
    var100 = []
    var500 = []

    for _ in range(n_to_average):
        my_swarm = Swarm(2, objfunc.func, n_particles = 10, start_range = [-5, 5], omega = a, c_1 = b, c_2 = b)
        for i in range(n_swarm_iters):
            my_swarm.update_particles()

            if i + 1 == 20:
                gbest20.append(get_error(my_swarm, objfunc))
                var20.append(np.var(my_swarm.current_pos))
            if i + 1 == 50:
                gbest50.append(get_error(my_swarm, objfunc))
                var50.append(np.var(my_swarm.current_pos))
            if i + 1 == 100:
                gbest100.append(get_error(my_swarm, objfunc))
                var100.append(np.var(my_swarm.current_pos))
            if i + 1 == 500:
                gbest500.append(get_error(my_swarm, objfunc))
                var500.append(np.var(my_swarm.current_pos))
    
    #save the variables
    returnObj.gbest20 = np.average(np.array(gbest20))
    returnObj.gbest50 = np.average(np.array(gbest50))
    returnObj.gbest100 = np.average(np.array(gbest100))
    returnObj.gbest500 = (np.average(np.array(gbest500)) if n_swarm_iters >= 500 else np.nan)

    returnObj.var20 = np.average(np.array(var20))
    returnObj.var50 = np.average(np.array(var50))
    returnObj.var100 = np.average(np.array(var100))
    returnObj.var500 = (np.average(np.array(var500)) if n_swarm_iters >= 500 else np.nan)

    return returnObj

class ConvergeTest():
    def __init__(self, a, b, objfunc, niter = 500, n_particles = 10, ):
        self.a = a
        self.b = b
        self.n_particles = n_particles
        self.objfunc = objfunc
        self.niter = niter

        self.gbest20 = 0
        self.gbest50 = 0
        self.gbest100 = 0
        self.gbest500 = 0
    
    def get_error(self, swarm):
        return (swarm.g_best-self.objfunc.minimum)@(swarm.g_best-self.objfunc.minimum)

    def numerical_investigate(self, deterministic = False):
        my_swarm = Swarm(2, self.objfunc.func, n_particles = self.n_particles, start_range = [-5, 5], omega = self.a, v_max = 0.1, c_1 = self.b, c_2 = self.b, deterministic = deterministic)
        
        for _ in range(20):
            for i in range(self.niter):
                my_swarm.update_particles()
                if i == 20:
                    self.gbest20 = self.get_error(my_swarm)
                    self.var20 = np.var(my_swarm.current_pos)
                if i == 50:
                    self.gbest50 = self.get_error(my_swarm)
                    self.var50 = np.var(my_swarm.current_pos)
                if i == 100:
                    self.gbest100 = self.get_error(my_swarm)
                    self.var100 = np.var(my_swarm.current_pos)
                if i == 500:
                    self.gbest500 = self.get_error(my_swarm)
                    self.var500 = np.var(my_swarm.current_pos)