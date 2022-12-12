import numpy as np
import random
import matplotlib.pyplot as plt

class Q_Learner():

    def __init__(self, env, n=2, m=15, alpha=0.05, beta=0.2, delta=0.95, sessions=1, log_frequency=10000):

        self.env = env
        self.n = n
        self.agents = [ 'agent_' + str(i) for i in range(n)]
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        #self.supervisor = supervisor
        #self.proportion_boost = proportion_boost
        #self.action_price_space = action_price_space
        self.sessions = sessions
        self.log_frequency = log_frequency

    def train(self):
        '''Train to fill q_table'''

        self.q_table = [{} for _ in range(self.n)]
        self.supervisor_q_table = {}

        # For plotting metrics
        all_rewards = [ [] for _ in range(self.n) ] # store the penalties per episode

        for i in range(self.sessions):    

            observation = self.env.reset()
            observation = str(observation['agent_0']) # Potentially make observation less long

            for agent in range(self.n):
                if observation not in self.q_table[agent]:
                    self.q_table[agent][observation] = [0] * self.m

            actions_dict_str = ''

            if actions_dict_str not in self.supervisor_q_table:
                self.supervisor_q_table[actions_dict_str] = [0] * self.n

            loop_count = 0
            reward_list = []
            done = False
            
            while not done:

                epsilon = np.exp(-1 * self.beta * loop_count)

                actions_dict = {}

                #Choose Action
                for agent in range(self.n):
                
                    if random.uniform(0, 1) < epsilon:
                        actions_dict[self.agents[agent]] = self.env.action_spaces[self.agents[agent]].sample()
                    else:
                        actions_dict[self.agents[agent]] = np.argmax(self.q_table[agent][observation])

                # Fix one of the policy
                #if random.uniform(0, 1) < epsilon:
                #    actions_dict[self.agents[0]] = self.env.action_spaces[self.agents[0]].sample()
                #else:
                #    actions_dict[self.agents[0]] = np.argmax(self.q_table[0][observation])
                #actions_dict[self.agents[1]] = 12

                # Step
                next_observation, reward, done, info = self.env.step(actions_dict)
                done = done['__all__']
                next_observation = str(next_observation['agent_0'])

 
                # Agent Update
                last_values = [0] * self.n
                Q_maxes = [0] * self.n

                for agent in range(self.n):
                    if next_observation not in self.q_table[agent]:
                        self.q_table[agent][next_observation] = [0] * self.m

                    last_values[agent] = self.q_table[agent][observation][actions_dict[self.agents[agent]]]
                    Q_maxes[agent] = np.max(self.q_table[agent][next_observation])
                
                    self.q_table[agent][observation][actions_dict[self.agents[agent]]] = ((1 - self.alpha) * last_values[agent]) + (self.alpha * (reward[self.agents[agent]] + self.delta * Q_maxes[agent]))

                reward_list.append(reward[self.agents[0]])

                observation = next_observation

                loop_count += 1

                if loop_count % self.log_frequency == 0:
                    mean_reward = np.mean(reward_list[-self.log_frequency:])
                    print(f"Session: {i}, \tLoop Count: {loop_count}, \t Epsilon: {epsilon}, \tMean Reward: {mean_reward}")
                
            mean_reward = np.mean(reward_list)

            print(f"Session: {i}, \tLoop Count: {loop_count}, \t Epsilon: {epsilon}, \tMean Reward: {mean_reward}")
            
            for agent in range(self.n):
                all_rewards[agent].append(mean_reward)

    def eval(self, observation, n=20):
        '''Eval q_table.'''

        observation = str(observation['agent_0'])

        for i in range(n):

            actions_dict = {}
            for agent in range(self.n):
                if observation not in self.q_table[agent]:
                    self.q_table[agent][observation] = [0] * self.m

                actions_dict[self.agents[agent]] = np.argmax(self.q_table[agent][observation])

            next_observation, reward, done, info = self.env.step(actions_dict)
            done = done['__all__']

            observation = str(next_observation['agent_0'])