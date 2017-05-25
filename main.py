import os.path
import math
import time
import numpy as np
from environment import Environment
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Activation, Flatten
from keras.optimizers import sgd, Adam

from experience_replay import ExperienceReplay


log_file = open('log.txt', 'w')
log_file.seek(0)

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

file_name = 'nagrade(-1,.1)_sa_odsecanjem_e1360.h5'
test = False

epsilon = .5
episodes = 10000
num_actions = 3
input_shape = 15
hidden_size = 60
max_memory = 2048
batch_size = 128

if os.path.isfile(file_name):
    print "citam"
    model = load_model(file_name)
    file_name = "nagrade(-1,.1)_sa_odsecanjem"
else:
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(input_shape,)))
    model.add(Activation('relu'))
    model.add(Dense(hidden_size))
    model.add(Activation('relu'))
    model.add(Dense(num_actions))
    adam = Adam()
    model.compile(loss='mean_squared_error', optimizer=adam)
# else end

env = Environment()
exp_replay = ExperienceReplay(max_memory=max_memory)
if os.path.isfile("memory.npy"):
    exp_replay.load_memory("memory")
can_play = env.reset()
time.sleep(1)
isnan = False
allReward = 0

for episode in range(1360, episodes):

    epsilon = (1.0 - (episode*1.0/episodes*1.0) ** .5) / 5.0

    loss = 0.0
    totalReward = 0
    can_play = env.reset()

    # waiting for game to start over
    while not can_play:
        can_play = env.reset()

    game_over = False
    state = env.get_state()
    start = time.time()
    while not game_over:
        state_p = state

        action_log = ""
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, num_actions)
            action_log = "Action: {} --- Random".format(action)
        else:
            q = model.predict(state_p.reshape((1, -1)))[0]
            action = np.argmax(q)
            action_log = "Action: {}".format(action)

        # print action_log
        state, reward, game_over = env.act(action)

        # store experience with probability 0.5 if there is no obstacle on screen
        if (np.random.rand() <= 0.05 and state_p[5] == -0.5) or state_p[5] != -0.5:
            exp_replay.remember((state_p, action, reward, state), game_over)
        #
        # # adapt model
        if exp_replay.can_learn():
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)
            loss = model.train_on_batch(inputs, targets)
            if math.isnan(targets[0][0]) or math.isnan(targets[1][0]) or math.isnan(targets[2][0]):
                isnan = True
                print "====================N A N============================="
                print inputs[0]
                print targets[0]
                print>> log_file, "====================N A N============================="
                print>> log_file, inputs[0]
                print>> log_file, targets[0]
                break

        totalReward += reward

    allReward += totalReward
    end = time.time()
    print "<<<Episode: {}; Total Reward: {}; loss: {}, Memory: {}; eps: {}, time: {}>>>" \
        .format(episode, totalReward, loss, exp_replay.memory_len(), epsilon, end - start)
    print>> log_file, "<<<Episode: {}; Total Reward: {}; loss: {}, Memory: {}; eps: {}, time: {}>>>" \
        .format(episode, totalReward, loss, exp_replay.memory_len(), epsilon, end - start)

    if episode % 20 == 0:
        model.save(file_name + "_e" + str(episode) + ".h5")  # creates a HDF5 file
        exp_replay.save_memory()
        avg = (allReward + 20) * 10 / 20.0
        print "===============Avg: {}".format(avg)
        print>> log_file, "===============Avg: {}".format(avg)
        allReward = 0

    if isnan:
        break


log_file.truncate()
log_file.close()
env.webdriver.close()
