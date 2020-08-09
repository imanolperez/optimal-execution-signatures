import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from esig import tosig
import itertools
import cvxpy as cp

from shuffle import shuffle

def find_dim(N):
    if N == 0:
        return 0
    if N == 1:
        return 3
    
    return tosig.sigdim(2, N)

def get_words(dim, order):
    keys = [tuple([t]) if isinstance(t, int) else t for t in map(eval, tosig.sigkeys(dim, order).split())]
    keys = [tuple(np.array(t) - 1) for t in keys]
    return keys

def build_problem(ES, q0, Lambda, k, phi, alpha, N, order):
    dim_l = find_dim(N)


    A = np.zeros((dim_l, dim_l))
    b = np.zeros(dim_l)
    c = 0.
    keys = get_words(2, order)

    alphabet = (0, 1)

    words = itertools.chain(*[itertools.product(alphabet, repeat=i) for i in range(N + 1)])

    for w in tqdm(words, total=find_dim(N)):
        w_idx = keys.index(w)

        words2 = itertools.chain(*[itertools.product(alphabet, repeat=i) for i in range(N + 1)])
        for v in words2:
            v_idx = keys.index(v)

            w_shuffle_v = shuffle(w, v)
            w_shuffle_v_1 = [tuple(list(tau) + [0]) for tau in w_shuffle_v]
            ES_w_shuffle_v_1 = sum(ES[tau] for tau in w_shuffle_v_1)
            
            w1_shuffle_v = shuffle(tuple(list(w) + [0]), v)
            ES_w1_shuffle_v = sum(ES[tau] for tau in w1_shuffle_v)


            w1_shuffle_v1 = shuffle(tuple(list(w) + [0]), tuple(list(v) + [0]))
            ES_w1_shuffle_v1 = sum(ES[tau] for tau in w1_shuffle_v1)

            w1_shuffle_v1_1 = [tuple(list(tau) + [0]) for tau in w1_shuffle_v1]
            ES_w1_shuffle_v1_1 = sum(ES[tau] for tau in w1_shuffle_v1_1)

            A[w_idx, v_idx] = -Lambda * ES_w_shuffle_v_1 + (k - alpha) * ES_w1_shuffle_v1 - (phi + k) * ES_w1_shuffle_v1_1



        ES_w = ES[w]

        w_shuffle_2 = shuffle(w, (1,))
        w_shuffle_21 = [tuple(list(tau) + [0]) for tau in w_shuffle_2]
        ES_w_shuffle_21 = sum(ES[tau] for tau in w_shuffle_21)

        w1_shuffle_2 = shuffle(tuple(list(w) + [0]), (1,))
        ES_w1_shuffle_2 = sum(ES[tau] for tau in w1_shuffle_2)

        w1 = tuple(list(w) + [0])
        ES_w1 = ES[w1]

        w11 = tuple(list(w1) + [0])
        ES_w11 = ES[w11]

        b[w_idx] = ES_w_shuffle_21 - ES_w1_shuffle_2 + (2 * alpha * q0 - q0 * k) * ES_w1 + 2 * phi * ES_w11

    c = q0 * (ES[(1,)] + 1.) - alpha * q0 - q0 * phi * ES[(1,)]
    
    return A, b, c


def optimise(ES, q0, Lambda, k, phi, alpha, N, order):
    A, b, c = build_problem(ES, q0, Lambda, k, phi, alpha, N, order)
    
    dim_l = find_dim(N)
    
    l = cp.Variable(dim_l)
    objective = cp.Maximize(cp.quad_form(l, A) + b * l)
    problem = cp.Problem(objective)
    problem.solve()
    
    return l.value