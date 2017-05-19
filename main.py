import os.path
import time
import numpy as np
from environment import Environment
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import sgd
from experience_replay import ExperienceReplay


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

file_name = '4_states_at_once_model'
test = True

epsilon = .5
episodes = 1000
num_actions = 3
input_shape = 13 * 4
hidden_size = input_shape
max_memory = 5000
batch_size = 200

# if os.path.isfile(file_name):
#     print "citam"
#     model = load_model(file_name)
# else:
model = Sequential()
model.add(Dense(hidden_size, input_shape=(input_shape,), activation='relu'))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(num_actions))
model.compile(sgd(lr=.01), "mse")
# else end

env = Environment()
exp_replay = ExperienceReplay(max_memory=max_memory)
can_play = env.reset()
time.sleep(2)

for episode in range(20, episodes):

    epsilon = 1.0 - episode * 1.0 / episodes*1.0
    if test:
        epsilon = 0

    loss = 0.0
    totalReward = 1
    can_play = env.reset()

    # waiting for game to start over
    while not can_play:
        can_play = env.reset()

    game_over = False
    state = env.get_state()

    while not game_over:
        state_p = state

        if np.random.rand() <= epsilon:
            action = np.random.randint(0, num_actions, size=1)
        else:
            q = model.predict(state_p.reshape((1, -1)))[0]
            action = np.argmax(q)

        state, reward, game_over = env.act(action)

        # store experience with probability 0.5 if there is no obstacle on screen
        if np.random.rand() <= 0.5 and \
           state_p[0*13 + 1] == -1 and \
           state_p[1*13 + 1] == -1 and \
           state_p[2*13 + 1] == -1 and \
           state_p[3*13 + 1] == -1:
            exp_replay.remember((state_p, action, reward, state), game_over)
        elif state_p[0*13 + 1] != -1 or \
           state_p[1*13 + 1] != -1 or \
           state_p[2*13 + 1] != -1 or  \
           state_p[3*13 + 1] != -1:
            exp_replay.remember((state_p, action, reward, state), game_over)

        # adapt model
        if exp_replay.can_learn():
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)
            loss += model.train_on_batch(inputs, targets)

        totalReward += reward
    print ("preskocio", totalReward, "ep", episode, "m", exp_replay.memory_len(), "e", epsilon)
    # if episode % 10 == 0:
    #     model.save(file_name + "_e" + str(episode) + ".h5")  # creates a HDF5 file 'my_model.h5'
    #     exp_replay.save_memory()

env.webdriver.close()
