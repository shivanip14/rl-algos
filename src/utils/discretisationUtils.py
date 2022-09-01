import numpy as np
from src.config.algoConfig import NO_OF_BINS

def discretiseStatesAndReturnBins(myEnv):
    state_size = myEnv.observation_space.shape[0]
    bins = []
    for state_feature_no in range(state_size):
        bins.append(np.linspace(myEnv.observation_space.high[state_feature_no], myEnv.observation_space.low[state_feature_no], NO_OF_BINS))
    return bins

def discretiseState(state, bins):
    state_size = len(bins)
    state_discretized = []
    for state_feature_no in range(state_size):
        for bin_no in range(NO_OF_BINS - 1):
            if bins[state_feature_no][bin_no] >= state[state_feature_no] > bins[state_feature_no][bin_no + 1]:
                state_discretized.append(bin_no)

    return tuple(state_discretized)