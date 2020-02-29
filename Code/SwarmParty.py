import numpy as np
import pylab as plt

class NN_Swarm():
    def __init__(self, n_particles = 30, x_max = 1, v_max = 0.1, c_1 = 2, c_2 = 2):
        self.n_particles = n_particles
        self.x_min = -1*x_max
        self.x_max = x_max
        self.v_min = -1*v_max
        self.v_max = v_max
        self.c_1 = c_1
        self.c_2 = c_2

    def provide_model(self, model):
        self.model = model
        self.ls_form = []
        my_weights = model.get_weights()
        index = 0
        for comp in my_weights:
            self.ls_form.append([index, index + np.size(comp), comp.shape])
            index += np.size(comp)
        converted_weights = self._Convert(my_weights)
        self.n_dims = converted_weights.shape[0]
        self.current_pos = self.x_min + (self.x_max-self.x_min)*np.random.rand(self.n_particles, self.n_dims) #initialise the particles
        self.p_best = self.current_pos
        self.v = self.v_min + (self.v_max-self.v_min)*np.random.rand(self.n_particles, self.n_dims)
    
    def provide_data(self, X_train, X_test, Y_train, Y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.g_best = self.p_best[np.argmin(self.f(self.current_pos)),:]
        self.pbest_perform = self.f(self.p_best)
        self.gbest_perform = self.f(self.g_best)
        self.nn_weights = self._ConvertBack(self.g_best)      

    def train(self, iterations = 50, give_curve = False, train_fast = False):
        """ Function to initiate swarm optimization of the weights and biases for the keras
        model provided. Runs for 50 iterations by default. Set give_curve = True to directly
        return a training curve (a numpy array with performance evaluations on training and 
        test data for each iteration) """

        try:
            self.model
            self.X_train
            self.X_test
            self.Y_train
            self.Y_test
        except AttributeError as error:
            print("\n\n***You must provide the swarm with both a model and training data before attempting to train***\n\n")
            print("Provide a model with: \n\n     your_swarm.provide_model(model)\n")
            print("Provide data with: \n\n     your_swarm.provide_data(X_train, X_test, Y_train, Y_test)\n")
            raise(error)

        #If it's asked for, provide a list to store the training curve
        if not train_fast:        
            training_curve = []
        is_better = np.zeros([self.n_particles,1])
        
        for i in range(iterations):
            m_v_max = self.v_max #*g_best_perform

            #update the positions using the velocity
            self.v += self.c_1*(np.random.rand(self.current_pos.shape[0],1))*(self.p_best-self.current_pos)
            self.v += self.c_2*(np.random.rand(self.current_pos.shape[0],1))*(self.g_best-self.current_pos)
            v_norm = np.linalg.norm(self.v,axis=1).reshape([self.v.shape[0],1])             #code for velocity limitation
            self.v = np.where(v_norm < m_v_max, self.v, m_v_max*self.v/v_norm)                       #(comment in to use it)
            self.current_pos += self.v
            curr_perform = self.f(self.current_pos)
            #replace the p_bests with the current location if they're better
            is_better = (curr_perform<self.pbest_perform).reshape([is_better.shape[0],1])
            self.p_best = is_better*self.current_pos + np.logical_not(is_better)*self.p_best
            self.pbest_perform = is_better*curr_perform + np.logical_not(is_better)*self.pbest_perform
            #update g_best
            if np.min(self.pbest_perform) < self.gbest_perform:
                self.g_best = self.p_best[np.argmin(self.pbest_perform),:]
                self.gbest_perform = np.min(self.pbest_perform)
            print("\rPerformance at iteration " + str(i+1) + ": " + str(self.gbest_perform), end = "\r")
            if not train_fast:
                training_curve.append([self.gbest_perform.astype(float), self.f(self.g_best, val_set = True)[0][0].astype(float)])
        self.nn_weights = self._ConvertBack(self.g_best) 
        if not train_fast:
            self.train_curve = np.array(training_curve)
            if give_curve:
                return np.array(training_curve)

    def f(self, positions, val_set = False):
        if positions.ndim == 1:
            positions = positions.reshape([1, positions.shape[0]])
        objective = np.zeros([positions.shape[0], 1])
        for part_ind, part_pos in enumerate(positions):
            x_1 = self._ConvertBack(part_pos)
            self.model.set_weights(x_1)
            if val_set == False:
                objective[part_ind] = self.model.evaluate(self.X_train, self.Y_train, verbose=0)
            else:
                objective[part_ind] = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
        return objective

    def plot_training(self):
        try:
            self.train_curve
        except AttributeError:
            print("No training curve available. Either training hasn't occured, or data recording was suppressed via the \"train_fast\" parameter")
            return
        plt.plot(self.train_curve[:,0], label="Training")
        plt.plot(self.train_curve[:,1], label="Validation")
        plt.legend()
        plt.show()

    def _Convert(self, pre_w):
        position = []
        for w in pre_w:
            position.append(w.flatten())
        return np.concatenate(position)
    
    def _ConvertBack(self, position):
        reinput = []
        for i in self.ls_form:
            reinput.append(position[i[0]:i[1]].reshape(i[2]))
        return reinput