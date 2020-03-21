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

def h_p(p):
    return p

class HqBeta:
    def __init__(self, lp_grad, beta):
        self.lp_grad=lp_grad
        self.beta=beta

    def __call__(self, q):
        return -self.beta*self.lp_grad(q)

def sample(flogprob, grad_logprob, q0, lp, last_grad, l, sc: SamplerState, beta=1):
    p = normal(size=q0.shape)
    current_k = kinetic(p)
    q = q0

    h_q = HqBeta(grad_logprob, beta)

    last_hq_q = -last_grad

    for i in range(l):
        q, p, last_hq_q = leapfrog(q, p, last_hq_q, sc.epsilon, h_q, h_p)

    current_u = -lp
    proposed_u = -flogprob(q)
    proposed_k = kinetic(p)
    accepted = False
    if uniform() < np.exp((current_u - proposed_u)*beta + current_k - proposed_k):
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


def sample_packed(args):
    return sample(*args)

def sample_pt_impl(flogprob, grad_logprob, q0_list, lp_list, last_grad_list, beta_list, l, sc: SamplerStatePt, executor=None):
    fmap=map if executor is None else executor.map
    nbeta=len(beta_list)
    n_per_beta=len(q0_list)//nbeta
    flogprob_beta=[flogprob]*len(q0_list)
    grad_beta=[grad_logprob]*len(q0_list)
    sc1=[SamplerState(sc.epsilon[i//n_per_beta], sc.target_accept_ratio, sc.adj_param) for i in range(len(q0_list))]
    expanded_beta_list=[beta_list[i//n_per_beta] for i in range(len(q0_list))]
    l_list=[l]*len(q0_list)
    result=list(fmap(sample_packed, zip(flogprob_beta, grad_beta, q0_list, lp_list, last_grad_list, l_list, sc1, expanded_beta_list)))
    #q0_proposed, lp_proposed, last_grad_proposed, accepted=[ [ x[i] for x in result] for i in range(4)]
    accept_cnt=[0]*nbeta
    for i,(q0, lp, gp, accepted) in enumerate(result):
        ibeta=i//n_per_beta
        if accepted:
            accept_cnt[ibeta]+=1
            q0_list[i]=q0
            lp_list[i]=lp
            last_grad_list[i]=gp
            if uniform() < 1 - sc.target_accept_ratio:
                sc.epsilon[ibeta] *= (1.0 + sc.adj_param)
        else:
            if uniform() < sc.target_accept_ratio:
                sc.epsilon[ibeta] /= (1.0 + sc.adj_param)
    return accept_cnt

def sample_pt(flogprob, grad_logprob, q0_list, lp_list, beta_list, l, sc: SamplerStatePt, executor=None):
    fmap=map if executor is None else executor.map
    last_grad_list=fmap(grad_logprob, q0_list)
    return sample_pt_impl(flogprob, grad_logprob, q0_list, lp_list, last_grad_list, beta_list, l, sc, executor)

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
