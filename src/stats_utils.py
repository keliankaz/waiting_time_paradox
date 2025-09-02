import numpy as np
import scipy

class Distribution:
    def __init__(self, dist, params):
        self.dist = dist
        self.params = params

    def sample(self):
        # Draw a sample from the distribution. To enforce a temporal point process,
        # we need to draw a sample until we get a positive value.
        while True: 
            t = self.dist(*self.params)
            if t > 0:
                return t
    
    def sample_interval(self, t1, t2):
        t = [t1]
        while True:
            t.append(t[-1] + self.sample())
            if t[-1] > t2:
                break
        return t[1:-1]
    
class WeibullDistribution(Distribution):
    def __init__(self, mean, shape=1):
        self.mean = mean
        self.params = (mean / scipy.special.gamma(1 + 1/shape), shape)

    # override the sample method
    def sample(self):
        return np.random.weibull(self.params[1]) * self.params[0]
 

def get_lapse_time(tq, t):
    dt = t - tq
    negative_dt = np.abs(dt[dt < 0])
    return np.min(negative_dt) if np.any(negative_dt) else np.nan

def get_waiting_time(tq, t):
    dt = t - tq
    positive_dt = dt[dt > 0]
    return np.min(positive_dt) if np.any(positive_dt) else np.nan

def get_full_interval(tq, t):
    return get_waiting_time(tq, t) + get_lapse_time(tq, t)