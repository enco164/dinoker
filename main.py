import os.path
import time
import numpy as np
from environment import Environment
from agent import Agent
from optparse import OptionParser


parser = OptionParser()
parser.add_option('-k', '--k', dest='k')
parser.add_option('-i', '--iterations', dest='iterations')
parser.add_option('-t', '--total_episodes', dest='total_episodes')
(options, args) = parser.parse_args()

k = int(options.k)
iterations = int(options.iterations)
total_episodes = int(options.total_episodes)

num_actions = 3
max_memory = 8192
batch_size = 64
hidden_size = 64
save_on_nth_episode = 10

log_file = open('log_e' + options.k + '.txt', 'w')
log_file.seek(0)

# fix random seed for reproducibility
seed = 164
np.random.seed(seed)

agent = Agent(num_actions=3,
              max_memory=max_memory,
              batch_size=batch_size,
              input_shape=15,
              hidden_size=hidden_size,
              network_file_name="network_e",
              memory_file_name="memory_e",
              iteration_postfix=str(k*iterations),
              log_file=log_file)

env = Environment()
can_play = env.reset()
time.sleep(1)
allReward = 0
mean_episode_time = 5.0
all_times = list()

if os.path.isfile('times' + options.k + '.npy'):
    all_times = np.load('times' + options.k + '.npy')
    all_times = all_times.tolist()

for episode in range(k * iterations, (k+1) * iterations + 1):

    episode_exploration_rate = 1 - episode*1.0 / total_episodes*1.0  # (-(episode*1.0 / episodes*1.0) ** 2) + 1
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
    times_played = 0
    while not game_over:
        state_p = state

        current_time = time.time()*1.0 - start_time
        exploration_rate = current_time / mean_episode_time * episode_exploration_rate  #((current_time / avg_episode_time) ** 2) * episode_exploration_rate

        # get action from agent based on state
        action = agent.get_action(state_p.reshape((1, -1)), exploration_rate)

        # send environment action
        state, reward, game_over = env.act(action)

        times_played += 1

        # store experience with probability 0.25 if there is no obstacle on screen
        if (np.random.rand() <= 0.25 and state_p[5] == -0.5) or state_p[5] != -0.5:
            agent.remember((state_p, action, reward, state), game_over)

        if reward > 0:
            totalReward += 1

    # adapt model
    for _ in range(times_played):
        agent.adapt_model()

    allReward += totalReward

    # store episode time length
    all_times.append(current_time)
    if len(all_times) > save_on_nth_episode:
        del all_times[0]

    # update mean_episode_time
    mean_episode_time = np.median(all_times)

    print "<<<Episode: {}; Total Reward: {}; eps: {}, E: {}, time: {}>>>" \
        .format(episode, totalReward, exploration_rate, episode_exploration_rate, current_time)
    print>> "{}, {}, {}, {}, {}" \
        .format(episode, totalReward, exploration_rate, episode_exploration_rate, current_time)

    if episode % save_on_nth_episode == 0:
        agent.save_model(postfix=str(episode))
        np.save("times" + "_e" + str(episode), all_times)
        avg = allReward*1.0 / save_on_nth_episode * 1.0
        print "===============Avg reward: {}".format(avg)
        allReward = 0


log_file.truncate()
log_file.close()
env.webdriver.close()
