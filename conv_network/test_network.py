import time
import numpy as np
from environment import Environment
from keras.models import load_model


file_name = './Conv2D_screenshot_e1520.h5'

episodes = 100000

model = load_model(file_name)

env = Environment()
can_play = env.reset()
time.sleep(1)
for episode in range(0, episodes):

    totalReward = 0
    can_play = env.reset()

    # waiting for game to start over
    while not can_play:
        can_play = env.reset()

    game_over = False
    state = env.get_state()

    while not game_over:
        q = model.predict(state)[0]

        action = np.argmax(q)

        state, reward, game_over = env.act(action)

        if reward > 0:
            totalReward += 1

    print "<<<Episode: {}; Total Reward: {}>>>"\
        .format(episode, totalReward)


env.webdriver.close()
