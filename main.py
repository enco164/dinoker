import os.path
import time
import numpy as np
from environment import Environment
from agent import Agent
from optparse import OptionParser


parser = OptionParser()
(options, args) = parser.parse_args()

log_file = open('log_e' + args[0] + '.txt', 'w')
log_file.seek(0)

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

episodes = 5000
num_actions = 3
max_memory = 4096
batch_size = 256
hidden_size = 64

save_on_nth_episode = 20

agent = Agent(num_actions=3,
              max_memory=max_memory,
              batch_size=batch_size,
              input_shape=15,
              hidden_size=hidden_size,
              network_file_name="network_e000_e" + args[0] + "00",
              memory_file_name="memory_e" + args[0] + "00",
              log_file=log_file)

env = Environment()
can_play = env.reset()
time.sleep(1)
isnan = False
allReward = 0

avg_episode_time = 5.0
all_times = list()

if os.path.isfile("all_times_e" + args[0] + "00" + '.npy'):
    all_times = np.load("all_times_e" + args[0] + "00" + '.npy')
    all_times = all_times.tolist()

r = int(args[0])
for episode in range(r * 100, (r+1) * 1000 * 2 + 1):

    episode_exploration_rate = 1 - episode*1.0 / episodes*1.0  # (-(episode*1.0 / episodes*1.0) ** 2) + 1
    loss = 0.0
    totalReward = 0
    can_play = env.reset()

    # waiting for game to start over
    while not can_play:
        can_play = env.reset()

    game_over = False
    state = env.get_state()

    start_time = time.time()
    current_time = time.time() * 1.0 - start_time
    while not game_over:
        state_p = state

        current_time = time.time()*1.0 - start_time
        exploration_rate = current_time / avg_episode_time * episode_exploration_rate  #((current_time / avg_episode_time) ** 2) * episode_exploration_rate

        # get action from agent based on state
        action = agent.get_action(state_p.reshape((1, -1)), exploration_rate)

        # send environment action
        state, reward, game_over = env.act(action)

        # store experience with probability 0.25 if there is no obstacle on screen
        if (np.random.rand() <= 0.25 and state_p[5] == -0.5) or state_p[5] != -0.5:
            agent.remember((state_p, action, reward, state), game_over)

        # adapt model
        agent.adapt_model()

        if reward > 0:
            totalReward += 1

    allReward += totalReward

    # store episode time length
    all_times.append(current_time)
    if len(all_times) > save_on_nth_episode:
        del all_times[0]

    # update avg_episode_time
    avg_episode_time = np.median(all_times)

    print "<<<Episode: {}; Total Reward: {}; eps: {}, E: {}, time: {}>>>" \
        .format(episode, totalReward, exploration_rate, episode_exploration_rate, current_time)
    print>> log_file, "<<<Episode: {}; Total Reward: {}; eps: {}, E: {}, time: {}>>>" \
        .format(episode, totalReward, exploration_rate, episode_exploration_rate, current_time)

    if episode % save_on_nth_episode == 0:
        agent.save_model(postfix="_e" + str(episode))
        np.save("all_times" + "_e" + str(episode), all_times)
        avg = allReward*1.0 / save_on_nth_episode * 1.0
        print "===============Avg reward: {}".format(avg)
        print>> log_file, "===============Avg reward: {}".format(avg)
        allReward = 0

    if isnan:
        break


log_file.truncate()
log_file.close()
env.webdriver.close()
