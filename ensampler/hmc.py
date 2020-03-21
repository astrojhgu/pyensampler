import scipy
import numpy as np
from numpy.random import normal, uniform


def leapfrog(q, p, last_hq_q, epsilon, h_q, h_p):
    p_mid = p - last_hq_q * epsilon / 2.0
    q = q + h_p(p_mid) * epsilon
    last_hq_q = h_q(q)
    p = p_mid - last_hq_q * (epsilon / 2.0)
    return (q, p, last_hq_q)


def kinetic(p, m=1):
    return np.dot(p, p) / m / 2


class SamplerState:
    def __init__(self, epsilon, target_accept_ratio, adj_param):
        self.epsilon = epsilon
        self.target_accept_ratio = target_accept_ratio
        self.adj_param = adj_param


def sample(flogprob, grad_logprob, q0, lp, last_grad, l, sc: SamplerState):
    p = normal(size=q0.shape)
    current_k = kinetic(p)
    q = q0
    h_p = lambda p: p
    h_q = lambda q: -grad_logprob(q)

    last_hq_q = -last_grad

    for i in range(l):
        q, p, last_hq_q = leapfrog(q, p, last_hq_q, sc.epsilon, h_q, h_p)

    current_u = -lp
    proposed_u = -flogprob(q)
    proposed_k = kinetic(p)
    accepted = False
    if uniform() < np.exp(current_u - proposed_u + current_k - proposed_k):
        q0 = q
        lp = -proposed_u
        last_grad = -last_hq_q
        accepted = True
        if uniform() < 1 - sc.target_accept_ratio:
            sc.epsilon *= 1.0 + sc.adj_param
    else:
        if uniform() < sc.target_accept_ratio:
            sc.epsilon /= (1.0 + sc.adj_param)

    return (q0, lp, last_grad, accepted)


class EvolveWrapper:
    def __init__(self, p_list,
                 q_list,last_hq_q_tmp,
                 n_per_beta,grad_logprob,beta_list, epsilon, l):
        self.p_list=p_list
        self.q_list=q_list
        self.last_hq_q_tmp=last_hq_q_tmp
        self.n_per_beta=n_per_beta
        self.grad_logprob=grad_logprob
        self.l=l
        self.beta_list=beta_list
        self.epsilon=epsilon

    def __call__(self, i):
        p1 = self.p_list[i]
        q1 = self.q_list[i]
        hlqq = self.last_hq_q_tmp[i]
        ibeta = i // self.n_per_beta
        beta = self.beta_list[ibeta]
        e = self.epsilon[ibeta]

        def h_p(p):
            return p

        def h_q(q):
            return self.grad_logprob(q) * (-beta)

        for j in range(self.l):
            q1, p1, hlqq = leapfrog(q1,
                                    p1,
                                    hlqq,
                                    e, h_q, h_p)
        return (q1, p1, hlqq)


class SamplerStatePt:
    def __init__(self, epsilon, target_accept_ratio, adj_param, nbeta: int):
        self.epsilon = epsilon * np.ones(nbeta)
        self.target_accept_ratio = target_accept_ratio
        self.adj_param = adj_param

    def freeze(self):
        self.adj_param=0.0

    def quick(self):
        self.adj_param=0.2

    def slow(self):
        self.adj_param=0.001

def sample_pt(flogprob, grad_logprob,
              q0_list, lp_list,
              beta_list, l, sc_list: SamplerStatePt,
              executor = None):
    map_func=map if executor is None else executor.map
    last_grad = list(map_func(gaussian_g, q0_list))
    return sample_pt_impl(flogprob, grad_logprob,
                   q0_list, lp_list, last_grad,
                   beta_list, l, sc_list, executor)

def sample_pt_impl(flogprob, grad_logprob,
                   q0_list, lp_list,
                   last_grad_list, beta_list,
                   l, sc_list: SamplerStatePt,
                   executor=None):
    nbeta = len(beta_list)
    accept_cnt = [0] * nbeta
    n_per_beta = len(q0_list) // nbeta
    assert (len(q0_list) == len(lp_list))
    assert (len(q0_list) == len(last_grad_list))
    assert (sc_list.epsilon.shape[0] == nbeta)
    map_func = map if executor is None else executor.map
    # p=normal(size=q0.shape)
    p_list = [np.random.normal(size=q0.shape) for q0 in q0_list]
    current_k = [kinetic(p) for p in p_list]
    q_list = q0_list.copy()

    last_hq_q_tmp = [-beta_list[i // n_per_beta] * x
                     for (i, x) in enumerate(last_grad_list)]

    evolve=EvolveWrapper(p_list, q_list,
                         last_hq_q_tmp, n_per_beta, grad_logprob,
                         beta_list, sc.epsilon.copy(),
                         l)
    qphqq_list = list(map_func(evolve, range(0, len(q_list))))
    q_list = [i[0] for i in qphqq_list]
    p_list = [i[1] for i in qphqq_list]
    last_hq_q_tmp = [i[2] for i in qphqq_list]
    current_u = [-x for x in lp_list]

    proposed_u = [-y for y in map_func(flogprob,
                          q_list)]
    proposed_k = [kinetic(p) for p in p_list]

    dh = [(u0 - u1) * beta_list[i // n_per_beta] + k0 - k1
          for (i, (u0, u1, k0, k1))
          in enumerate(zip(
            current_u, proposed_u, current_k, proposed_k))]
    for i, (dh1, q1, pu, lhqt) in enumerate(
            zip(
                dh, q_list, proposed_u, last_hq_q_tmp)):
        ibeta = i // n_per_beta
        if np.isfinite(dh1) and np.random.rand() < np.exp(dh1):
            q0_list[i] = q1
            lp_list[i] = -pu
            last_grad_list[i] = -lhqt
            accept_cnt[ibeta] += 1
            if np.random.rand() < 1.0 - sc.target_accept_ratio:
                sc.epsilon[ibeta] *= (1.0 + sc.adj_param)
        elif np.random.rand() < sc.target_accept_ratio:
            sc.epsilon[ibeta] /= (1.0 + sc.adj_param)
    return accept_cnt


def rosenbrock(x):
    return -np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def delta(i, j):
    return 1 if i == j else 0


def diff_rosenbrock(x):
    result = np.zeros_like(x)
    for j in range(len(x)):
        for i in range(len(x) - 1):
            result[j] -= 200.0 * (x[i + 1] - x[i] ** 2) * (delta(j, i + 1) - 2 * x[i] * delta(i, j)) + 2.0 * (
                        x[i] - 1) * delta(i, j)

    return result


def gaussian(x):
    return -np.sum(x ** 2) / 2


def gaussian_g(x):
    return -x


if __name__ == '__main__':
    from multiprocessing import Pool
    import sys
    pool=Pool()
    x = [np.array([0., 0.])] * 16
    lp = list(map(gaussian, x))
    last_grad = list(map(gaussian_g, x))
    beta_list = [1.0, 0.25]
    sc = SamplerStatePt(0.01, 0.9, 0.0001, len(beta_list))
    sc.quick()
    # sys.exit()
    of = open('a.txt', 'w')

    for i in range(0, 100000):
        ac = sample_pt_impl(gaussian, gaussian_g,
                            x, lp, last_grad,
                            beta_list, 3, sc,
                            executor=pool)

        if i % 100==0 and i>10000:
            print(sc.epsilon)
            sc.freeze()
            of.write("{0} {1} {2} {3}\n".format(
                x[0][0], x[0][1], x[8][0], x[8][1]))
