import os.path
import math
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Activation, Conv2D, Flatten, Dense
from keras.optimizers import Adam
from experience_replay import ExperienceReplay


class Agent(object):
    def __init__(self, num_actions,
                 max_memory=2048, batch_size=128, network_file_name='conv_network',
                 memory_file_name="memory", log_file=open('log.txt', 'w'),
                 iteration_postfix=""):

        self.log_file = log_file
        self.num_actions = num_actions
        self.network_file_name = network_file_name
        self.batch_size = batch_size

        self.exp_replay = ExperienceReplay(max_memory=max_memory,
                                           memory_file_name=memory_file_name,
                                           iteration_postfix=iteration_postfix)

        if os.path.isfile(network_file_name + iteration_postfix + '.h5'):
            print "===Reading model==="
            self.model = load_model(network_file_name + iteration_postfix + '.h5')
        else:
            print "===Creating model==="
            model = Sequential()
            model.add(Conv2D(16, (8, 8), padding="same", strides=(4, 4), input_shape=(84, 84, 4)))
            model.add(Activation('relu'))
            model.add(Conv2D(32, (4, 4), padding="same", strides=(2, 2)))
            model.add(Activation('relu'))
            model.add(Flatten())
            model.add(Dense(256))
            model.add(Activation('relu'))
            model.add(Dense(num_actions))
            model.add(Activation('linear'))
            adam = Adam()
            model.compile(loss='mean_squared_error', optimizer=adam)
            self.model = model

    def get_action(self, state, exploration_rate):
        if np.random.rand() <= exploration_rate:
            action = np.random.randint(0, self.num_actions)
            
        else:
            q = self.model.predict(state)[0]
            action = np.argmax(q)
        
        return action

    def remember(self, states, game_over):
        self.exp_replay.remember(states, game_over)

    def adapt_model(self):
        if self.exp_replay.can_learn():
            inputs, targets = self.exp_replay.get_batch(self.model, batch_size=self.batch_size)
            self.model.train_on_batch(inputs, targets)

            if math.isnan(targets[0][0]) or math.isnan(targets[0][1]) or math.isnan(targets[0][2]):
                print "====================N A N============================="
                print inputs[0]
                print targets[0]
                print>> self.log_file, "====================N A N============================="
                print>> self.log_file, inputs[0]
                print>> self.log_file, targets[0]

    def save_model(self, postfix=""):
        self.model.save(self.network_file_name + postfix + ".h5")  # creates a HDF5 file
        self.exp_replay.save_memory(postfix)
