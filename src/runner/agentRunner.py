from src.environments.environmentSetter import SetAndReturnEnvironment
from src.config.runnerConfig import MAX_TRAINING_TRIALS
from tqdm import tqdm

def Run(algo):
    myEnv = SetAndReturnEnvironment()
    episodic_reward = 0
    max_reward = float('-inf')

    for episode in tqdm(range(MAX_TRAINING_TRIALS), desc='Running training trials'):
        done = False
        myEnv.reset()
        episodic_reward = 0
        while not done:
            myEnv.render()
            obs, reward, done, info = myEnv.step(GetActionForAlgo(algo, myEnv))
            episodic_reward += reward
        print('\nEpisode {} resulted in {} reward'.format(episode, episodic_reward))
        if episodic_reward > max_reward:
            max_reward = episodic_reward
            print('\nNew max reward achieved: {}'.format(max_reward))

    myEnv.close()

def GetActionForAlgo(algo, myEnv):
    # Default random agent taking a random action
    action = myEnv.action_space.sample()
    # TBD other algos
    return action