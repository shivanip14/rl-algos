import numpy as np
from random import *
from src.config.algoConfig import NO_OF_BINS, DYNA_SIMULATIONS
from src.utils.discretisationUtils import discretiseStatesAndReturnBins, discretiseState
import logging
import sys

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

class DynaQLearning:

    def __init__(self, myEnv):
        # Only discrete actions allowed as of now
        # TODO allow continuous action spaces
        self.num_of_actions = myEnv.action_space.n

        # Discretise states
        # TODO not to discretise if they're already Discrete
        self.bins = discretiseStatesAndReturnBins(myEnv)
        self.per_state_size = myEnv.observation_space.shape[0]
        self.Q_table = np.zeros([NO_OF_BINS] * self.per_state_size + [self.num_of_actions])

        # The model-based part of Dyna-Q
        self.T_dict = {}
        self.model_dict = {}
        self.s_a_count = np.zeros([NO_OF_BINS] * self.per_state_size + [self.num_of_actions])


        logging.info('Q-Table shape [{}]'.format(self.Q_table.shape))
        self.epsilon = 0.05
        self.alpha = 0.15
        self.gamma = 0.95

    def getAction(self, myEnv, currentState):
        explorationProbability = random()
        if explorationProbability >= self.epsilon:
            return myEnv.action_space.sample()
        else:
            currentState_discrete = discretiseState(currentState, self.bins)
            return np.argmax(self.Q_table[currentState_discrete])

    def executeAction(self, myEnv, currentState, action):
        nextState, immediateReward, done, info = myEnv.step(action)

        currentState_discrete = discretiseState(currentState, self.bins)
        nextState_discrete = discretiseState(nextState, self.bins)

        # Update Q-table according to Bellman equation
        self.Q_table[currentState_discrete][action] = self.Q_table[currentState_discrete][action] + (self.alpha * (immediateReward + self.gamma * np.max(self.Q_table[nextState_discrete]) - self.Q_table[currentState_discrete][action]))

        ##### Dyna start #####
        self.s_a_count[currentState_discrete][action] += 1
        self.model_dict[currentState_discrete + tuple([action])] = (nextState_discrete, immediateReward)


        for simulation_no in range(DYNA_SIMULATIONS):
            simCurrentState_discrete, simAction, simImmediateReward, simNextState_discrete = self.simulateExperience(myEnv)
            # Update Q table with sim exp
            self.Q_table[simCurrentState_discrete][simAction] = self.Q_table[simCurrentState_discrete][simAction] + (self.alpha * (simImmediateReward + self.gamma * np.max(self.Q_table[simNextState_discrete]) - self.Q_table[simCurrentState_discrete][simAction]))

            # Don't set the next state as current state - imp difference from the Q-learning iteration!

        ##### Dyna end #####

        return nextState, immediateReward, done, info

    def simulateExperience(self, myEnv):
        newRandomState_discrete, newRandomAction = self.getPreviouslySeenRandomStateAndAction()
        # Get simNextState_discrete & simReward from (newRandomState_discrete, newRandomAction) from using model/T+R
        key = tuple(newRandomState_discrete) + tuple([newRandomAction])
        simNextState_discrete = self.model_dict[key][0]
        simReward = self.model_dict[key][1]
        return newRandomState_discrete, newRandomAction, simReward, simNextState_discrete

    def getPreviouslySeenRandomStateAndAction(self):
        random_seen_state_action = choice(np.argwhere(self.s_a_count > 0)) # Using boolean masking here - cool stuff!
        random_seen_state = random_seen_state_action[:-1]
        random_seen_action = random_seen_state_action[-1]
        return random_seen_state, random_seen_action

    def resetAgent(self, myEnv):
        myEnv.reset()

    def getQTable(self):
        return self.Q_table