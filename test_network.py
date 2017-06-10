import time
import numpy as np
from environment import Environment
from keras.models import load_model


file_name = 'network_e3000.h5'
test = False

episodes = 100000

model = load_model(file_name)

env = Environment()
can_play = env.reset()
time.sleep(1)
for episode in range(0, episodes):

    loss = 0.0
    totalReward = 0
    can_play = env.reset()

    # waiting for game to start over
    while not can_play:
        can_play = env.reset()

    game_over = False
    state = env.get_state()

    while not game_over:
        start = time.time()
        q = model.predict(state.reshape((1, -1)))[0]
        # print q

        action = np.argmax(q)

        state, reward, game_over = env.act(action)
        end = time.time()
        if reward > 0:
            totalReward += 1
        # print "Action: {}; Time: {}".format(action, end-start)
    print "<<<Episode: {}; Total Reward: {}>>>"\
        .format(episode, totalReward)


env.webdriver.close()
