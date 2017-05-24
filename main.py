import os.path
import math
import time
import numpy as np
from environment_images import EnvironmentImages
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Activation, Flatten
from keras.optimizers import Adam
from experience_replay import ExperienceReplay


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

log_file = open('log.txt', 'w')
log_file.seek(0)

file_name = 'Conv2D_screenshot_e990.h5'
test = False

epsilon = .5
episodes = 100000
num_actions = 3
input_shape = 15
hidden_size = 100
max_memory = 300
batch_size = 20

if os.path.isfile(file_name):
    print "citam"
    model = load_model(file_name)
else:
    model = Sequential()
    model.add(Conv2D(32, (8, 8), padding="same", strides=(4, 4), input_shape=(84, 84, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), padding="same", strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding="same", strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(3))

    adam = Adam(lr=.05)
    model.compile(loss='mse', optimizer=adam)
# else end

file_name = 'Conv2D_screenshot'

env = EnvironmentImages()
exp_replay = ExperienceReplay(max_memory=max_memory)
exp_replay.load_memory("memory")
can_play = env.reset()
time.sleep(1)
isnan = False
for episode in range(990, episodes):

    epsilon = 1.0 - (episode*1.0/episodes*1.0) ** .5
    if test:
        epsilon = 0

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

        # action_log = ""
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, num_actions)
            # action_log = "Action: {} --- Random".format(action)
        else:
            q = model.predict(state_p)[0]
            action = np.argmax(q)
            # action_log = "Action: {}".format(action)

        # print action_log
        # start = time.time()
        state, reward, game_over = env.act(action)

        if env.get_obstacle_pos() != 1000 and np.random.rand() <= .4:
            exp_replay.remember((state_p, action, reward, state), game_over)

        # # adapt model
        if exp_replay.can_learn():
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)
            loss += model.train_on_batch(inputs, targets)

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
    end = time.time()
    print "<<<Episode: {}; Total Reward: {}; Memory: {}; eps: {}, time: {}>>>"\
        .format(episode, totalReward, exp_replay.memory_len(), epsilon, end - start)
    print>> log_file, "<<<Episode: {}; Total Reward: {}; Memory: {}; eps: {}, time: {}>>>"\
        .format(episode, totalReward, exp_replay.memory_len(), epsilon, end - start)
    if episode % 20 == 0:
        model.save(file_name + "_e" + str(episode) + ".h5")  # creates a HDF5 file
        exp_replay.save_memory()
    if isnan:
        break

log_file.truncate()
log_file.close()

env.webdriver.close()
