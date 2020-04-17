import numpy as np
import pylab as plt
import tensorflow as tf
from tensorflow import keras
import wandb

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
    
    def provide_data(self, X_train, X_test, Y_train, Y_test, names = None):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.names = names
        print(self.current_pos.shape)
        try:
            float(self.model.evaluate(self.X_test, self.Y_test, verbose=0))
        except TypeError as error:
            print("Model.evaluate is returning multiple values.\nThis is likely because you have asked the model to return additional metrics.\nPlease remove the \'metrics=\' part of model.compile")
            raise(error)
        self.g_best = self.p_best[np.argmin(self.f(self.current_pos)),:]
        self.pbest_perform = self.f(self.p_best)
        self.gbest_perform = self.f(self.g_best)
        self.nn_weights = self._ConvertBack(self.g_best)      

    def train(self, num_epochs = 30, give_curve = False, train_fast = False, patience = False):
        """ Function to initiate swarm optimization of the weights and biases for the keras
        model provided. Runs for 50 num_epochs by default. Set give_curve = True to directly
        return a training curve (a numpy array with performance evaluations on training and 
        test data for each iteration) """

        try:
            self.model.set_weights(self._ConvertBack(self.g_best))
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
        
        for epoch in range(num_epochs):
            
            ## PSO Loop
            m_v_max = self.v_max*np.exp(epoch/num_epochs) #*g_best_perform

            #update the positions using the velocity
            self.v += self.c_1*(np.random.rand(self.current_pos.shape[0],1))*(self.p_best-self.current_pos)
            self.v += self.c_2*(np.random.rand(self.current_pos.shape[0],1))*(self.g_best-self.current_pos)
            v_norm = np.linalg.norm(self.v,axis=1).reshape([self.v.shape[0],1])             #code for velocity limitation
            self.v = np.where(v_norm < m_v_max, self.v, m_v_max*self.v/v_norm)                       #(comment in to use it)
            self.current_pos += self.v
            curr_perform = self.f(self.current_pos)
            
            #replace the p_bests with the current location if they are better
            is_better = (curr_perform<self.pbest_perform).reshape([is_better.shape[0],1])
            self.p_best = is_better*self.current_pos + np.logical_not(is_better)*self.p_best
            self.pbest_perform = is_better*curr_perform + np.logical_not(is_better)*self.pbest_perform
            
            #update g_best
            if np.min(self.pbest_perform) < self.gbest_perform:
                self.g_best = self.p_best[np.argmin(self.pbest_perform),:]
                self.gbest_perform = np.min(self.pbest_perform)
                count_no_improv = 0
            elif patience != False:
                count_no_improv += 1
                if count_no_improv > patience:
                    break
            print("\rPerformance at iteration " + str(epoch+1) + ": " + str(self.gbest_perform), end = "\r")

            # log wandb parameters
            wandb.log({'epoch': epoch, 'train_loss': self.gbest_perform.astype(float), 'val_loss':self.f(self.g_best, val_set = True)[0][0].astype(float)})
            
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
    
    def get_accuracy(self, give_curve = False, isMnist = False):
        
        if isMnist == True:
            # set weights and predict with model
            probability_model = keras.Sequential([self.model, keras.layers.Softmax()])
            predictions = probability_model.predict(self.X_test)

            def plot_image(i, predictions_array, true_label, img):
                predictions_array, true_label, img = predictions_array, true_label[i], img[i]
                plt.grid(False)
                plt.xticks([])
                plt.yticks([])

                plt.imshow(img, cmap=plt.cm.binary)

                predicted_label = np.argmax(predictions_array)
                if predicted_label == true_label:
                    color = 'blue'
                else:
                    color = 'red'

                plt.xlabel("{} {:2.0f}% ({})".format(names[predicted_label],
                                            100*np.max(predictions_array),
                                            names[true_label]),
                                            color=color)

            def plot_value_array(i, predictions_array, true_label):
                predictions_array, true_label = predictions_array, true_label[i]
                plt.grid(False)
                plt.xticks(range(10))
                plt.yticks([])
                thisplot = plt.bar(range(10), predictions_array, color="#777777")
                plt.ylim([0, 1])
                predicted_label = np.argmax(predictions_array)

                thisplot[predicted_label].set_color('red')
                thisplot[true_label].set_color('blue')
            
            if give_curve == True:
                num_rows = 5
                num_cols = 3
                num_images = num_rows*num_cols
                plt.figure(figsize=(2*2*num_cols, 2*num_rows))
                for i in range(num_images):
                    plt.subplot(num_rows, 2*num_cols, 2*i+1)
                    plot_image(i, predictions[i], self.Y_test, self.X_test)
                    plt.subplot(num_rows, 2*num_cols, 2*i+2)
                    plot_value_array(i, predictions[i], self.Y_test)
                plt.tight_layout()
                plt.show()

            NN_output_args = np.argmax(predictions, axis = 1)
            real_output_args = self.Y_test

            test_accuracy = tf.keras.metrics.Accuracy()
            test_accuracy(NN_output_args, real_output_args)
        
        else:
            # set weights and predict with model
            self.model.set_weights(self.nn_weights)
            predictions = self.model.predict(self.X_test)

            NN_output_args = np.argmax(predictions, axis = 1)
            real_output_args = np.argmax(self.Y_test, axis = 1)

            if give_curve == True:
                plt.figure(figsize = (10,1))
                plt.plot(NN_output_args,'ro', label = "NN")
                plt.plot(real_output_args, 'bo', label = "Real")
                plt.legend()

            test_accuracy = tf.keras.metrics.Accuracy()
            test_accuracy(NN_output_args, real_output_args)      
        
        return test_accuracy.result()


    def plot_training(self, figName = None):
        try:
            self.train_curve
        except AttributeError:
            print("No training curve available. Either training hasn't occured, or data recording was suppressed via the \"train_fast\" parameter")
            return
        plt.plot(self.train_curve[:,0], '-', label="Training")
        plt.plot(self.train_curve[:,1], '--', label="Validation")
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()
        
        if figName != None:
            plt.savefig(str(figName), bbox_inches="tight")

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


class Swarm():
    
    def __init__(self, n_dims, function, n_particles = 30, start_range = [-1, 1], v_max = 0.1, c_1 = 2, c_2 = 2, omega = 1):
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
    
    def f(self, positions):
        return self.function(positions)

    def update_particles(self):
        """ Function to update particle positions using the PSO algorithm"""

        #exponential velocity decay
        # m_v_max = self.v_max#*np.exp(epoch/num_epochs)

        #update the current velocity
        self.v *= self.omega
        self.v += self.c_1*(np.random.rand(self.current_pos.shape[0],1))*(self.p_best-self.current_pos)
        self.v += self.c_2*(np.random.rand(self.current_pos.shape[0],1))*(self.g_best-self.current_pos)
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