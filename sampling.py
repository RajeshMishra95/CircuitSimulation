import numpy as np
import random

class Sample(object):
    """
    Defines the given vector as the probability distribution
    from which a sample is picked.
    """
    def __init__(self, distribution):
        self.distribution = distribution

    def __oneNorm(self):
        '''Calculates the 1-norm of a given vector.'''
        one_norm = float(0)
        for i in self.distribution:
            one_norm += abs(i)
        return one_norm

    def probDistribution(self):
        '''Converts the quasiprobability distribution to a probability
        distribution.'''
        prob = np.zeros(len(self.distribution))
        for i in range(len(self.distribution)):
            prob[i] = abs(self.distribution[i])/self.__oneNorm()
        return prob

    def index(self):
        '''Samples a value from the distribution.'''
        cdf = np.zeros(len(self.distribution)+1)
        for i in range(len(self.distribution)):
            cdf[i+1] = cdf[i]+self.probDistribution()[i]
        
        random_value = random.random()
        idx = 0

        for i in range(len(cdf)):
            if random_value <= cdf[i]:
                idx = i-1
                break
        return idx
