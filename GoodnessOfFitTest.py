import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import math as mth
from statsmodels.distributions.empirical_distribution import ECDF
import ditr as dstrb
from scipy.stats import norm, lognorm, gamma, laplace, logistic, wald, rayleigh, cauchy, t, johnsonsu, rice

class GoodnessOfFitTest(object):
    def __init__(self):
        self.name = "Base"
    
    def statistic(self, x, dist=None):
        raise NotImplementedError("Подклассам Goodness_of_fit_test необходимо определить метод 'statistic'.")
    
class Kolmogorov(GoodnessOfFitTest):
    name = "Kolmogorov"
    def __init__(self):
        self.mean = 0
        self.sigma = 1
    
    def statistic(self, x, dist=None, est=None):
        n = len(x)
        y = np.sort(x)
        F = dist.func(y, est[0], est[1])
        
        def calc_d_plus(F):
            d = np.array([(index[0] + 1) / n - value for index, value in np.ndenumerate(F)])
            return max(d)

        def calc_d_minus(F):
            d = np.array([value - index[0] / n for index, value in np.ndenumerate(F)])
            return max(d)

        def calc_d_n(F):
            d_min = calc_d_minus(F)
            d_plus = calc_d_plus(F)
            return max(d_min, d_plus)
        
        d_n = calc_d_n(F)
        s_star = (6 * n * d_n + 1) / (6 * mth.sqrt(n))
        return s_star

class KramerVonMizesSmirnov(GoodnessOfFitTest):
    name = "KramerVonMizesSmirnov"
    def __init__(self):
        self.mean = 0
        self.sigma = 1
    
    def statistic(self, x, dist=None, est=None):
        n = len(x)
        y = np.sort(x)
        F = dist.func(y, est[0], est[1])
        s_star = np.array([(value - (2 * (index[0] + 1) - 1) / (2 * n)) ** 2 for index, value in np.ndenumerate(F)])
        s_star = 1 / (12 * n) + np.sum(s_star)
        return s_star

class AndersonDarling(GoodnessOfFitTest):
    name = "AndersonDarling"
    def __init__(self):
        #self.mean = 0
        self.sigma = 1
    
    def statistic(self, x, dist=None, est=None):
        n = len(x)
        y = np.sort(x)
        F = dist.func(y, est[0], est[1])
        s_star = np.array([(2 * (index[0] + 1) - 1) * np.log(value) / (2 * n) + \
            (1 - (2 * (index[0] + 1) - 1) / (2 * n)) * np.log(1 - value) \
            for index, value in np.ndenumerate(F)])
        s_star = -n - 2 * np.sum(s_star)
        return s_star
    

class Tests(dstrb.Change_Distribution):
    name = "GOFTests"
    def __init__(self, sample, n, dic_dist):
        self._criterions = []
        self.sample = sample
        self.n = n
        self.dic_dist = dic_dist

    def add_criterion(self, h):
        self._criterions.append(h)

    def test(self):
        chg_d = dstrb.Change_Distribution()
        chg_d.add_distr(self.dic_dist)
        def gen_s_n(chg_d, M=100):
            #s_star = np.array([self._criterions[0][0].statistic(self.sample, dist=d, est=[d.mu, d.sigma]) for d in chg_d._distributions[0]])
            s_star = np.array([[c.statistic(self.sample, dist=d, est=[d.mu, d.sigma]) for d in chg_d._distributions[0]] for c in self._criterions[0]])
            sample_mod = np.array([[h.gen_sample(self.n, h.mu, h.sigma) for i in range(M)] for h in chg_d._distributions[0]])
            est_n_sample = np.array([[chg_d._distributions[0][i].get_est_n_sample(sample_mod[i,j]) for j in range(M)] \
                                    for i in range(len(chg_d._distributions[0]))])
            #s_n = np.array([[self._criterions[0][0].statistic(sample_mod[i,j], dist=chg_d._distributions[0][i], est=est_n_sample[i,j]) for j in range(M)]\
            #                   for i in range(len(chg_d._distributions[0]))]) \
            s_n = np.array([[[c.statistic(sample_mod[i,j], dist=chg_d._distributions[0][i], est=est_n_sample[i,j]) for j in range(M)]\
                            for i in range(len(chg_d._distributions[0]))] \
            for c in self._criterions[0]])
            return s_star, s_n

        def calc_p_values(chg_d, s_n, s_star):
            ecdf_s_n = np.array([[ECDF(sn[i]) for i in range(len(chg_d._distributions[0]))] for sn in s_n])
            p_value = np.array([[(1 - ecdf(sst)) for ecdf, sst in zip(ecdf_s_n[c], s_star[c])] for c in range(len(self._criterions[0]))])
            return p_value

        sst, sn = gen_s_n(chg_d)
        pv = calc_p_values(chg_d, sn, sst)
        return sst, pv

r = norm.rvs(loc=100., scale=2, size=500)
dictnr = [dstrb.Normal(mode=1, sample=r), dstrb.Wald(mode=1, sample=r)]
tst = Tests(r, len(r), dictnr)
tst.add_criterion([Kolmogorov(), KramerVonMizesSmirnov(), AndersonDarling()])
sst, pv = tst.test()
