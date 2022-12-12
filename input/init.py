"""
Model of algorithms and competition
"""
from gym.spaces import Discrete, Box
import numpy as np
from itertools import product
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from ray.rllib.env import MultiAgentEnv

class model(MultiAgentEnv):
    """
    model

    Attributes
    ----------
    n : int
        number of players
    alpha : float
        product differentiation parameter
    beta : float
        exploration parameter
    delta : float
        discount factor
    mu : float
        product differentiation parameter
    a : int
        value of the products
    a0 : float
        value of the outside option
    c : float
        marginal cost
    k : int
        dimension of the grid
    stable: int
        periods of game stability
    """

    def __init__(
            self,
            n = 2,
            c = 1,
            a=2,
            a0 = 0,
            mu = 0.25,
            #delta = 0.95,
            m = 15,
            xi = 0.1,
            k = 1, 
            max_steps = 200, 
            sessions = 1, 
            convergence = 5, 
            trainer_choice = 'Q_Learner', 
            path='', 
            savefile=''
        ):
        """Initialize game with default values"""
        super(model, self).__init__()
        # Default properties
        self.n = n

        # Length of Memory
        self.k = k

        # Marginal Cost
        self.c = c

        # Number of Discrete Prices
        self.m = m

        # Product Quality Indexes
        self.a = a

        # Product Quality Index: Outside Good
        self.a0 = a0

        # Index of Horizontal Differentiation
        self.mu = mu

        self.agent_idx = 0

            # MultiAgentEnv Action and Observation Space
        self.agents = ['agent_' + str(i) for i in range(n)]
        self.observation_spaces = {}
        self.action_spaces = {}

        if k > 0:
            self.numeric_low = np.array([0] * (k * n))
            numeric_high = np.array([m] * (k * n))
            obs_space = Box(self.numeric_low, numeric_high, dtype=int)
        else:
            self.numeric_low = np.array([0] * n)
            numeric_high = np.array([m] * n)
            obs_space = Box(self.numeric_low, numeric_high, dtype=int)

        for agent in self.agents:
            self.observation_spaces[agent] = obs_space
            self.action_spaces[agent] = Discrete(m)



        self.pN = self.nash_sol()
        self.pM = self.monopoly_sol()
        self.action_price_space = np.linspace(self.pN - xi * (self.pM - self.pN), self.pM + xi * (self.pM - self.pN), m)
        self.reward_range = (-float('inf'), float('inf'))
        self.current_step = None
        self.max_steps = max_steps
        self.sessions = sessions
        self.convergence = convergence
        self.convergence_counter = 0
        self.trainer_choice = trainer_choice
        self.action_history = {}
        self.path = path
        self.savefile = savefile


        for agent in self.agents:
            if agent not in self.action_history:
                self.action_history[agent] = [self.action_spaces[agent].sample()]


        self.reset()
    
    #def demand_monopoly(self, p):
    #    """Computes demand"""
    #    e = np.exp((self.a - p) / self.mu)
    #    d = e / (e + np.exp(self.a0 / self.mu))
    #    return d
    
    def demand(self, P, agent):
        ''' Demand as a function of product quality indexes, price, and mu. '''
        a = np.array([self.a] * self.n)
        #P = [p] * self.n
        return np.exp((a[agent] - P[agent])/self.mu) /  (np.sum(np.exp((a- P) / self.mu)) + np.exp(self.a0 / self.mu))


    def foc(self, p):
        ''' Derivative for demand function '''
        a = np.array([self.a] * self.n)
        #P = [p] * self.n
        denominator = np.exp(self.a0 / self.mu)
        for i in range(self.n):
            denominator += np.exp((a[i] - p[i]) / self.mu)
        function_list = []
        for i in range(self.n):
            term = np.exp((a[i] - p[i]) / self.mu)
            first_term = term / denominator
            second_term = (np.exp((2 * (a[i] - p[i])) /  self.mu) * (-self.c + p[i])) / ((denominator ** 2) * self.mu)
            third_term = (term * (-self.c + p[i])) / (denominator * self.mu)
            function_list.append((p[i] - self.c) *0.5*(first_term + second_term - third_term))
        return function_list

    #def foc(self, p):
    #    """Compute first order condition"""
    #    d = self.demand(p)
    #    zero = 1 - (p - self.c) * (1 - d) / self.mu
    #    return np.squeeze(zero)

    #def foc_monopoly(self, p):
    #    """Compute first order condition of a monopolist"""
    #    d = self.demand(p)
    #    d1 = np.flip(d)
    #    p1 = np.flip(p)
    #    zero = 1 - (p - self.c) * (1 - d) / self.mu + (p1 - self.c) * d1 / self.mu
    #    return np.squeeze(zero)

        # Monopoly Equilibrium Price
    def monopoly_func(self, p):
        return -(p[self.agent_idx]- self.c) * self.demand(p,self.agent_idx)


    #def compute_p_competitive_monopoly(self):
    #    """Computes competitive and monopoly prices"""
    #    p0 = np.ones((1, self.n)) * 3 * self.c
    #    p_competitive = fsolve(self.foc, p0)
    #    p_monopoly = fsolve(self.foc_monopoly, p0)
    #    return p_competitive, p_monopoly

    def nash_sol(self):
        nash_sol = optimize.root(self.foc, [2] * self.n) # args is the initial guess: #!CRUCIAL
        pN = nash_sol.x[0]
        return pN

    def monopoly_sol(self):
        monopoly_sol = optimize.minimize(self.monopoly_func, 2) #initial guess not crucial
        pM = monopoly_sol.x[0]
        return pM



    def step(self, actions_dict):
        ''' MultiAgentEnv Step '''

        actions_idx = np.array(list(actions_dict.values())).flatten()

        for i in range(self.n):
            self.action_history[self.agents[i]].append(actions_idx[i])

        if self.k > 0:
            obs_agents = np.array([self.action_history[self.agents[i]][-self.k:] for i in range(self.n)], dtype=object).flatten()
            observation = dict(zip(self.agents, [obs_agents for i in range(self.n)]))
        else:
            observation = dict(zip(self.agents, [self.numeric_low for _ in range(self.n)]))

        self.prices = self.action_price_space.take(actions_idx[:self.n])
        reward = np.array([0.0] * self.n)


        for i in range(self.n):
            reward[i] = (self.prices[i] - self.c) * self.demand( self.prices,  i)
            # reward[i] = (self.prices[i] - self.c_i) * demand[i]

        reward = dict(zip(self.agents, reward))

        if self.action_history[self.agents[0]][-2] == self.action_history[self.agents[0]][-1]:
            self.convergence_counter += 1
        else:
            self.convergence_counter = 0

        if self.convergence_counter == self.convergence or self.current_step == self.max_steps:
            done = {'__all__': True}
        else:
            done = {'__all__': False}

        info = dict(zip(self.agents, [{} for _ in range(self.n)]))


        self.current_step += 1

        return observation, reward, done, info

    def one_step(self):
        step_actions_dict = {}

        for agent in self.agents:
            step_actions_dict[agent] = self.action_history[agent][-1]


        observation, _, _, _ = self.step(step_actions_dict)

        return observation

    def deviate(self, direction='down'):
        deviate_actions_dict = {}

        if direction == 'down':
            # First agent deviates to lowest price
            deviate_actions_dict[self.agents[0]] = 3
        elif direction == 'up':
            # First agent deviates to highest price
            deviate_actions_dict[self.agents[0]] = self.m - 3

        for agent in range(1, self.n):
            # All other agents remain at previous price (large assumption)
            deviate_actions_dict[self.agents[agent]] = self.action_history[self.agents[agent]][-1]


        observation, _, _, _ = self.step(deviate_actions_dict)

        return observation

    def reset(self):
        self.current_step = 0

        # Reset to random action
        random_action = np.random.randint(self.m, size=self.n)

        for i in range(random_action.size):
            self.action_history[self.agents[i]].append(random_action[i])

        if self.k > 0:
            obs_agents = np.array([self.action_history[self.agents[i]][-self.k:] for i in range(self.n)], dtype=object).flatten()
            observation = dict(zip(self.agents, [obs_agents for i in range(self.n)]))
        else:
            observation = dict(zip(self.agents, [self.numeric_low for _ in range(self.n)]))

            
        return observation

    def plot(self, window=1000, overwrite_id=0):
        '''Plot action history.'''
        warnings.filterwarnings('ignore')
        n = len(self.action_history[self.agents[0]])
        x = np.arange(n)
        for agent in self.agents:
            plt.plot(x, self.action_price_space.take(self.action_history[agent]), alpha=0.75, label=agent)
        for agent in self.agents:
            plt.plot(x, pd.Series(self.action_price_space.take(self.action_history[agent])).rolling(window=window).mean(), alpha=0.5, label=agent + ' MA')
        plt.plot(x, np.repeat(self.pM, n), 'r--', label='Monopoly')
        plt.plot(x, np.repeat(self.pN, n), 'b--', label='Nash')
        plt.xlabel('Steps')
        plt.ylabel('Price')
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        # plt.title(self.savefile.replace('_', ' ').title())
        plt.legend(loc='upper left')
        plt.savefig('./figures/' + self.savefile + '_' + str(overwrite_id))
        #plt.clf()

    def plot_last(self, last_n=1000, window=None, title_str = '', overwrite_id=0):
        '''Plot action history.'''
        x = np.arange(last_n)
        for agent in self.agents:
            plt.plot(x, self.action_price_space.take(self.action_history[agent][-last_n:]), alpha=0.75, label=agent)
        if window is not None:
            for agent in self.agents:
                plt.plot(x, pd.Series(self.action_price_space.take(self.action_history[agent][-last_n:])).rolling(window=window).mean(), alpha=0.5, label=agent + ' MA')
        plt.plot(x, np.repeat(self.pM, last_n), 'r--', label='Monopoly')
        plt.plot(x, np.repeat(self.pN, last_n), 'b--', label='Nash')
        plt.xlabel('Steps')
        plt.ylabel('Price')
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        # plt.title((self.savefile + title_str + ' Eval ' + str(last_n) ).replace('_', ' ').title())
        plt.legend()
        plt.savefig('./figures/' + self.savefile + title_str + '_eval_' + str(last_n) + '_' + str(overwrite_id))
        #plt.clf()

    def render(self, mode='human'):
        raise NotImplementedError