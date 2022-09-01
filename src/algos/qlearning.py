import numpy as np
from random import random
from src.config.algoConfig import NO_OF_BINS
from src.utils.discretisationUtils import discretiseStatesAndReturnBins, discretiseState
import logging
import sys

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

class QLearning:

    def __init__(self, myEnv):
        # Only discrete actions allowed as of now
        # TODO allow continuous action spaces
        self.num_of_actions = myEnv.action_space.n

        # Discretise states
        self.bins = discretiseStatesAndReturnBins(myEnv)
        self.per_state_size = myEnv.observation_space.shape[0]
        self.Q_table = np.zeros([NO_OF_BINS] * self.per_state_size + [self.num_of_actions]) # WOHOOOOOO!

        logging.info('Q-Table shape [{}]'.format(self.Q_table.shape))
        logging.info('Q-table non-zero entries at indices: {}'.format(np.nonzero(self.Q_table)))
        self.epsilon = 0.05
        self.alpha = 0.1
        self.gamma = 0.95

    def getAction(self, myEnv, currentState):
        explorationProbability = random()
        if explorationProbability >= self.epsilon:
            return myEnv.action_space.sample()
        else:
            currentState_discrete = discretiseState(currentState, self.bins)
            return np.argmax(self.Q_table[currentState_discrete])

    def executeAction(self, myEnv, currentState, action):
        nextState, immediate_reward, done, info = myEnv.step(action)

        currentState_discrete = discretiseState(currentState, self.bins)
        nextState_discrete = discretiseState(nextState, self.bins)

        # Update Q-table according to Bellman equation
        self.Q_table[currentState_discrete][action] = self.Q_table[currentState_discrete][action] + (self.alpha * (immediate_reward + self.gamma * np.max(self.Q_table[nextState_discrete]) - self.Q_table[currentState_discrete][action]))

        if done:
            self.resetAgent(myEnv)
        return nextState, immediate_reward, done, info

    def resetAgent(self, myEnv):
        myEnv.reset()
        self.Q_table = np.zeros([NO_OF_BINS] * self.per_state_size + [self.num_of_actions])

    def getQTable(self):
        return self.Q_table