from src.environments.environmentSetter import SetAndReturnEnvironment
from src.config.runnerConfig import MAX_TRAINING_TRIALS
from tqdm import tqdm
from src.algos.qlearning import QLearning
from src.algos.dynaqlearning import DynaQLearning
import logging, sys

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

def run(algoName):
    myEnv = SetAndReturnEnvironment()
    episodic_reward = 0
    max_reward = float('-inf')

    algo = None
    tabular = False
    if algoName == 'qlearning':
        algo = QLearning(myEnv)
        tabular = True
    elif algoName == 'dynaq':
        algo = DynaQLearning(myEnv)
        tabular = True
    # TODO other algos

    for episode in tqdm(range(MAX_TRAINING_TRIALS), desc='Running training trials'):
        done = False
        currentState = myEnv.reset()
        episodic_reward = 0
        while not done:
            myEnv.render()
            action = getActionForAlgo(algo, myEnv, currentState)
            nextState, reward, done, info = executeActionForAlgo(algo, myEnv, currentState, action)
            currentState = nextState
            episodic_reward += reward
        #logging.debug('Episode {} resulted in {} reward'.format(episode, episodic_reward))
        if episodic_reward > max_reward:
            max_reward = episodic_reward
            #logging.debug('New max reward achieved: {}'.format(max_reward))

    logging.info('Max reward achieved in training: {}'.format(max_reward))
    myEnv.close()

def getActionForAlgo(algo, myEnv, currentState):
    # Default random agent taking a random action
    if not algo:
        action = myEnv.action_space.sample()
        #logging.debug('Not using any particular algorithm, choosing action {} at random'.format(action))
    else:
        action = algo.getAction(myEnv, currentState)
        #logging.debug('Action return by algo object: {}'.format(action))
    return action

def executeActionForAlgo(algo, myEnv, currentState, action):
    if not algo:
        return myEnv.step(action)
    else:
        return algo.executeAction(myEnv, currentState, action)