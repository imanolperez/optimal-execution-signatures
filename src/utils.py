import numpy as np
from tqdm.auto import tqdm
import iisignature

def transform(X):
    if len(X) == 1:
        return np.array([[-X[0, 0], X[0, 1]], [X[0, 0], X[0, 1]]])
    new_X = [[-X[1, 0], X[0, 1]]]
    for x_past, x_future in zip(X[:-1], X[1:]):
        new_X.append(x_past)
        new_X.append([x_past[0], x_future[1]])
        
    new_X.append(X[-1])
    
    return np.array(new_X)

def Cost(path, speed, q0, Lambda, k, phi, alpha, **kwargs):
    WT = 0.
    QT = q0
    L2_penalty = 0.
    delta_t = path[1, 0] - path[0, 0]
    permanent_impact = 0.
    for i in range(len(path)):
        speed_t = speed(np.array(path[:i + 1]))
        permanent_impact += k * speed_t * delta_t
        
        temporary_impact = Lambda * speed_t
        
        WT += (path[i, 1] - permanent_impact - temporary_impact) * speed_t * delta_t
        QT -= speed_t * delta_t
        L2_penalty += QT**2 * delta_t

    C = WT + QT * (path[-1, 1] - permanent_impact - alpha * QT) - phi * L2_penalty
    
    return C
    

def sig_speed(l, N):
    def f(path):
        path = transform(path)
        if N == 0:
            sig = np.array([1.])
        elif N == 1:
            sig = np.array([1., path[-1, 0] - path[0, 0], path[-1, 1] - path[0, 1]])
        else:
            sig = np.r_[1., iisignature.sig(path, N)]

        return sig.dot(l)
    
    return f


def get_words(dim, order):
    keys = [tuple([t]) if isinstance(t, int) else t for t in map(eval, tosig.sigkeys(dim, order).split())]
    keys = [tuple(np.array(t) - 1) for t in keys]
    return keys


def get_analytics(speed, sample, Lambda=None, q0=None, alpha=None, k=None, **kwargs):

    paths_Qt = []
    paths_wealth = []
    paths_speed = []

    for path in tqdm(sample):
        WT = 0.
        Qt = [0.]
        speeds = []
        permanent_impact = 0.
        for i in range(len(path) - 1):
            delta_t = path[i + 1, 0] - path[i, 0]
            speed_t = speed(np.array(path[:i + 1]))
            permanent_impact += k * speed_t * delta_t

            temporary_impact = Lambda * speed_t

            WT += (path[i, 1] - permanent_impact - temporary_impact) * speed_t * delta_t
            Qt.append(Qt[-1] + speed_t * delta_t)
            speeds.append(speed_t)
        
        Qt = q0 - np.array(Qt)
        WT += Qt[-1] * (path[-1, 1] - permanent_impact  - alpha * Qt[-1])
        paths_Qt.append(Qt)
        paths_wealth.append(WT)
        paths_speed.append(speeds)

        
    return paths_speed, paths_Qt, paths_wealth