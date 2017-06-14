import numpy as np
import os
import time


class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.9, memory_file_name="memory", iteration_postfix=""):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount
        self.memory_file_name = memory_file_name
        self.save_called = 0

        # load memory
        if os.path.isfile(memory_file_name + iteration_postfix + '.npy'):
            self.load_memory(memory_file_name + iteration_postfix + '.npy')

    def remember(self, states, game_over):
        # memory[i] = ((state_t, action_t, reward_t, state_t+1), game_over?)
        self.memory.append((states, game_over))
        while len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape
        # init empty arrays
        inputs = np.zeros((min(len_memory, batch_size), env_dim[1], env_dim[2], env_dim[3]))
        targets = np.zeros((inputs.shape[0], num_actions))

        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            # read from memory
            ((state_t, action_t, reward_t, state_tp1), game_over) = self.memory[idx]

            # fill first state
            inputs[i:i + 1] = state_t

            # Calculate values for all actions,.
            # Thou shalt not correct actions not taken #deep
            targets[i] = model.predict(state_t)[0]
            Q_sa = np.max(model.predict(state_tp1)[0])

            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa

        return inputs, targets

    def save_memory(self, postfix=""):
        if self.save_called % 4 == 0:
            np.save(self.memory_file_name + postfix, self.memory)
        self.save_called = self.save_called + 1

    def load_memory(self, file_name):
        self.memory = np.load(file_name)
        self.memory = self.memory.tolist()

    def can_learn(self):
        return len(self.memory) > 32

    def memory_len(self):
        return len(self.memory)
