import os
import sys
import gym
import torch
from collections import deque
from copy import deepcopy
from dqn_agent import DQNAgent
from utils import *


if __name__ == "__main__":
    EPISODES = 500000
    HEIGHT = 84
    WIDTH = 84
    HISTORY_SIZE = 4
    PLOT_FOLDER = 'save_graph'
    MODEL_FOLDER = 'save_model'
    plot_path, model_path = check_dirs_exist(PLOT_FOLDER, MODEL_FOLDER)

    env = gym.make('BreakoutDeterministic-v4')
    env.reset()
    state_size = env.observation_space.shape
    action_size = env.action_space.n - 1

    scores, episodes = [], []
    agent = DQNAgent(action_size, HISTORY_SIZE)
    recent_reward = deque(maxlen=100)
    frame = 0

    for e in range(EPISODES):
        done = False
        score = 0

        history = np.zeros([5, 84, 84], dtype=np.uint8)
        step = 0
        state = env.reset()

        history = get_init_state(history, state, HISTORY_SIZE, HEIGHT, WIDTH)

        while not done:
            step += 1
            frame += 1
            if agent.render:
                env.render()

            action = agent.get_action(np.float32(history[:4, :, :]) / 255.)

            next_state, reward, done, info = env.step(action+1)

            pre_proc_next_state = preprocess(next_state, HEIGHT, WIDTH)
            history[4, :, :] = pre_proc_next_state

            r = np.clip(reward, -1, 1)

            agent.append_sample(deepcopy(pre_proc_next_state), action, r, done)

            if frame >= agent.train_start:
                agent.train_model(frame)
                if frame % agent.update_target == 0:
                    agent.update_target_model()

            score += reward
            history[:4, :, :] = history[1:, :, :]

            if frame % 50000 == 0:
                scores.append(score)
                episodes.append(e)
                save_plot(episodes, scores, plot_path)
                print(f'Saved plot to {plot_path}')

            if done:
                recent_reward.append(score)
                print(f"episode: {e}, score: {score}, memory length: {len(agent.memory)}, "
                      f"epsilon: {agent.epsilon}, steps: {step}, recent reward: {np.mean(recent_reward)}")

                # if the mean of scores of last 10 episode is bigger than 400, stop training
                if np.mean(recent_reward) > 50:
                    torch.save(agent.model, os.path.join(model_path, 'breakout_dqn'))
                    sys.exit()
