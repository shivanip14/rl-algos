import gym

def SetAndReturnEnvironment():
    myEnv = gym.make("CartPole-v0")
    print('Environment: {}'.format(myEnv.env))
    print('Action space: {}'.format(myEnv.action_space))
    print('Observation space: {}'.format(myEnv.observation_space))
    return myEnv
