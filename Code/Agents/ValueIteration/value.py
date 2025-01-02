import json
import random
import highway_env
import gymnasium as gym

import numpy as np
from collections import defaultdict
import os
from tqdm import tqdm


import sys
sys.path.append(os.path.abspath('..'))
from metrics import Metrics

# %load_ext tensorboard
import sys
sys.path.insert(0, '/content/HighwayEnv/scripts/')


class QLearningAgent:
    def __init__(self, env, params):
        self.env = env
 
        self.exploration_rate = params.get("exploration_rate", 0.3)
        self.q_table = defaultdict()
        self.q_table_path = "q_table.json"
        self.load_q_table()
        self.action_space = env.action_space.n
        use_metrics = params.get("use_metrics", False)

        self.discount_factor = params.get("gamma", 0.9) # Discount Factor
        self.episode_num = params.get("episode_num", 100)
        self.metrics = Metrics("value_iteration", "training_results", use_metrics)


    def choose_action(self, state):
        if random.random() < self.exploration_rate:
            return self.env.action_space.sample()  # Explore: random action
        else:
            try:
                return str(max(self.q_table[state], key = self.q_table[state].get))
            except:
                return self.env.action_space.sample()


    def train(self):
        for epoch in tqdm(range(self.episode_num), desc="Training Model"):
            state = str(self.env.reset()[0])  # Convert state to string for indexing
            done = False
            theta = 0.01
            gamma_counter = 0
            truncated = False
            episode_rewards = []
            while not done and not truncated:

                action = str(self.choose_action(state))

                next_obs, reward, done, truncated, info = self.env.step(action)
                next_state = str(next_obs)

                if state not in self.q_table:
                    self.q_table[state] = {str(i): 0 for i in range(0, self.action_space)}


                if next_state not in self.q_table:
                    self.q_table[next_state] = {str(i): 0 for i in range(0, self.action_space)}

                best_next_action = str(max(self.q_table[next_state], key = self.q_table[next_state].get))
                
                self.q_table[state][action] = reward + self.discount_factor * self.q_table[next_state][best_next_action]

                episode_rewards.append(reward + self.discount_factor * self.q_table[next_state][best_next_action])


                gamma_counter += 1

                state = next_state

                
                if (self.discount_factor** gamma_counter * reward ) < theta:
                    break


            self.metrics.add("rollout/rewards", sum(episode_rewards) / len(episode_rewards), epoch)
            self.metrics.add("rollout/episode-length", gamma_counter, epoch)

            if (gamma_counter + 1) % 50 == 0:
                print(f"Episode {gamma_counter + 1}/{self.episode_num}")


        self.save_q_table()
        self.metrics.close()

    def evaluate(self, episodes = 10):
        for episode in tqdm(range(episodes), desc="Evaluate Model"):
            state = str(self.env.reset()[0])
            done = False
            total_reward = 0
            step = 0
            
            while not done:
                step += 1
                
                try:
                    action = str(max(self.q_table[state], key = self.q_table[state].get))
                except:
                    action = self.env.action_space.sample()

                next_obs, reward, done, truncated, info = self.env.step(action)
                state = str(next_obs)
                total_reward += reward
                if total_reward > 50:
                    break
                # self.env.render()

            # print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Steps: {step}")

    def load_q_table(self):
        if os.path.exists(self.q_table_path):
            try:
                with open(self.q_table_path, 'r') as file:
                # self.q_table = np.load(self.q_table_path, allow_pickle=True).item()
                    loaded =  json.load(file)
                    self.q_table.update(loaded)
                    # print(type(self.q_table))
                    print("Q-table loaded successfully.")
            except Exception as e:
                print(f"Error loading Q-table: {e}")

    def save_q_table(self):
        try:
            with open(self.q_table_path, 'w') as file:  
                json.dump(self.q_table, file, indent=4)
                print("Q-table saved successfully.")
        except Exception as e:
            print(f"Error saving Q-table: {e}")


config = {
    "lanes_count": 3,
    "observation": {
        "type": "TimeToCollision",
        "horizon": 5,
    }}

env = gym.make("highway-fast-v0", render_mode="rgb_array", config=config)

params = {
    "use_metrics": True,
    "episode_num": 10,
    "gamma": 0.9, # Discount Factor
    "exploration_rate": 0.3,
}

agent = QLearningAgent(env, params=params)
agent.train()