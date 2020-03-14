import numpy as np
import numpy.random as random
np.set_printoptions(precision=3)


def scale_vec(x1, x2, z):
    return np.array(x1)*z+np.array(x2)*(1-z)

def propose_move(p1, p2, z):
    return scale_vec(p1, p2, z)

def draw_z(a):
    sqrt_a=np.sqrt(a)
    p=random.uniform(0, 2.*(sqrt_a-1./sqrt_a))
    y=1.0/sqrt_a+p/2.0
    return y**2

def exchange_prob(lp1, lp2, beta1, beta2):
    return np.exp((beta2-beta1)*(lp1-lp2))

def swap_walkers(ensemble, logprob, beta_list):
    nbeta=len(beta_list)
    nwalkers_per_beta=len(ensemble)//nbeta
    j1list=list(range(nwalkers_per_beta))
    j2list = list(range(nwalkers_per_beta))
    for i in range(1, nbeta):
        beta1=beta_list[i]
        beta2=beta_list[i-1]
        random.shuffle(j1list)
        lp1_list=np.array([logprob[i*nwalkers_per_beta+j1] for j1 in j1list])
        lp2_list=np.array([logprob[(i-1)*nwalkers_per_beta+j2] for j2 in j2list])
        ep=exchange_prob(lp1_list, lp2_list, beta1, beta2)
        masks=random.rand(nwalkers_per_beta)<ep
        for m, j1, j2 in zip(masks, j1list, j2list):
            if m:
                n1=i*nwalkers_per_beta+j1
                n2=(i-1)*nwalkers_per_beta+j2
                ensemble[n1], ensemble[n2]=ensemble[n2], ensemble[n1]
                logprob[n1], logprob[n2]=logprob[n2], logprob[n1]




def sample_pt(flogprob, ensemble, cached_logprob, a=2.0, beta_list=[1.0], mpi_executor=None):
    nwalkers=len(ensemble)
    nbetas=len(beta_list)
    nwalkers_per_betas=nwalkers//nbetas
    ndim=len(ensemble[0])
    assert(nwalkers_per_betas*nbetas==nwalkers)
    assert (nwalkers>0)
    assert (nwalkers_per_betas%2==0)
    assert (len(ensemble)==len(cached_logprob))

    pair_id=[]
    for ibeta in range(nbetas):
        offset=ibeta*nwalkers_per_betas
        b=[i+offset for i in range(nwalkers_per_betas)]
        random.shuffle(b)
        pair_id+=(list(zip(b[0::2], b[1::2]))+list(zip(b[1::2], b[0::2])))
    pair_id.sort()
    #print("")
    #print(ensemble)
    #print(pair_id)
    z_list=[draw_z(a) for i in cached_logprob]
    #print(z_list)
    proposed_pt=[propose_move(ensemble[i1], ensemble[i2], z) for ((i1, i2), z) in zip(pair_id, z_list)]
    #print(z_list)
    #print(proposed_pt)
    if mpi_executor is None:
        #new_logprob=np.array([ flogprob(pt) for pt in proposed_pt ])
        new_logprob=np.array(list(map(flogprob, proposed_pt)))
    else:
        new_logprob=np.array(list(mpi_executor.map(flogprob, proposed_pt)))
    delta_lp=new_logprob-np.array(cached_logprob)
    expanded_beta=np.repeat(beta_list, nwalkers_per_betas)
    q_list=np.exp((ndim-1)*np.log(z_list)+delta_lp*expanded_beta)
    #print(q_list)
    mask=random.rand(nwalkers)<q_list
    next_ensemble=[a if m else b for (m, a, b) in zip(mask, proposed_pt, ensemble)]
    next_lp = [a if m else b for (m, a, b) in zip(mask, new_logprob, cached_logprob)]
    return next_ensemble, next_lp

