""" Модуль распределений

Модуль содержит распределения функций для аппроксимаций распределений средних и дисперсий
Использован паттер - цепочка обязанностей
"""

import numpy as np
from scipy.stats import norm, lognorm, gamma, laplace, logistic, wald, rayleigh, cauchy, t, johnsonsu, rice
import matplotlib.pyplot as plt
import scipy as sp
import math as mth

class Distribution(object):
    def __init__(self):
        self.name = "Base"
        self.mean = 0
        self.sigma = 1
    
    def density(self, x):
        raise NotImplemented("Подклассам Distribution необходимо определить метод 'density'.")
    
    def func(self, x):
        raise NotImplemented("Подклассам Distribution необходимо определить метод 'func'.")

    def gen_sample(self, n):
        raise NotImplemented("Подклассам Distribution необходимо определить метод 'gen_sample'.")

#for mu distribution x [-inf;+inf]
class Normal(Distribution):
    name = "Normal"
    def __init__(self, mode=0, elem=None, sample=None):
        if mode == 0:
            self.mu = elem[0]
            self.sigma = elem[1]
        else:
            self.mu, self.sigma = norm.fit(sample)
        self.math_average = norm.mean(loc=self.mu, scale=self.sigma)
        self.dispersion = norm.var(loc=self.mu, scale=self.sigma)
    
    def density(self, x, _mu=None, _sigma=None):
        return norm.pdf(x, loc=_mu, scale=_sigma)    

    def func(self, x, _mu=None, _sigma=None):
        return norm.cdf(x, loc=_mu, scale=_sigma)

    def gen_sample(self, n, _mu=None, _sigma=None):
        return norm.rvs(loc=_mu, scale=_sigma, size=n)

    def get_est_n_sample(self, sample):
        _mu, _sigma = norm.fit(sample)
        return _mu, _sigma 

class Logistic(Distribution):
    name = "Logistic"
    def __init__(self, mode=0, elem=None, sample=None):
        if mode == 0:
            self.mu = elem[0]
            self.sigma = elem[1]
        else:
            self.mu, self.sigma = logistic.fit(sample)
        self.math_average = logistic.mean(loc=self.mu, scale=self.sigma)
        self.dispersion = logistic.var(loc=self.mu, scale=self.sigma)
    
    def density(self, x):
        return logistic.pdf(x, loc=self.mu, scale=self.sigma)    

    def func(self, x):
        return logistic.cdf(x, loc=self.mu, scale=self.sigma)

    def gen_sample(self, n):
        return logistic.rvs(loc=self.mu, scale=self.sigma, size=n)

class Laplace(Distribution):
    name = "Laplace"
    def __init__(self, mode=0, elem=None, sample=None):
        if mode == 0:
            self.mu = elem[0]
            self.sigma = elem[1]
        else:
            self.mu, self.sigma = laplace.fit(sample)
        self.math_average = laplace.mean(loc=self.mu, scale=self.sigma)
        self.dispersion = laplace.var(loc=self.mu, scale=self.sigma)
    
    def density(self, x):
        return laplace.pdf(x, loc=self.mu, scale=self.sigma)    

    def func(self, x):
        return laplace.cdf(x, loc=self.mu, scale=self.sigma) 

    def gen_sample(self, n):
        return laplace.rvs(loc=self.mu, scale=self.sigma, size=n)

class Cauchy(Distribution):
    name = "Cauchy"
    def __init__(self, mode=0, elem=None, sample=None):
        if mode == 0:
            self.mu = elem[0]
            self.sigma = elem[1]
        else:
            self.mu, self.sigma = cauchy.fit(sample)
        self.math_average = cauchy.mean(loc=self.mu, scale=self.sigma)
        self.dispersion = cauchy.var(loc=self.mu, scale=self.sigma)
    
    def density(self, x):
        return cauchy.pdf(x, loc=self.mu, scale=self.sigma)    

    def func(self, x):
        return cauchy.cdf(x, loc=self.mu, scale=self.sigma) 

    def gen_sample(self, n):
        return cauchy.rvs(loc=self.mu, scale=self.sigma, size=n)

class Students_t(Distribution):
    name = "Student's t"
    def __init__(self, mode=0, elem=None, sample=None):
        if mode == 0:
            self.df = elem[0]
            self.mu = elem[1]
            self.sigma = elem[2]
        else:
            self.df, self.mu, self.sigma = t.fit(sample)
        self.math_average = t.mean(self.df, loc=self.mu, scale=self.sigma)
        self.dispersion = t.var(self.df, loc=self.mu, scale=self.sigma)
    
    def density(self, x):
        return t.pdf(x, self.df, loc=self.mu, scale=self.sigma)    

    def func(self, x):
        return t.cdf(x, self.df, loc=self.mu, scale=self.sigma)

    def gen_sample(self, n):
        return t.rvs(self.df, loc=self.mu, scale=self.sigma, size=n)

#
#
#
#for sigma distribution x [0;+inf]
class Gamma(Distribution):
    name = "Gamma"
    def __init__(self, mode=0, elem=None, sample=None):
        if mode == 0:
            self.a = elem[0]
            self.mu = elem[1]
            self.sigma = elem[2]
        else:
            self.a, self.mu, self.sigma = gamma.fit(sample)
        self.math_average = gamma.mean(self.a, loc=self.mu, scale=self.sigma)
        self.dispersion = gamma.var(self.a, loc=self.mu, scale=self.sigma)
    
    def density(self, x):
        return gamma.pdf(x, self.a, loc=self.mu, scale=self.sigma)    

    def func(self, x):
        return gamma.cdf(x, self.a, loc=self.mu, scale=self.sigma)

    def gen_sample(self, n):
        return gamma.rvs(self.a, loc=self.mu, scale=self.sigma, size=n)

class Lognormal(Distribution):
    name = "Lognormal"
    def __init__(self, mode=0, elem=None, sample=None):
        if mode == 0:
            self.s = elem[0]
            self.mu = elem[1]
            self.sigma = elem[2]
        else:
            self.s, self.mu, self.sigma = lognorm.fit(sample)
        self.math_average = lognorm.mean(self.s, loc=self.mu, scale=self.sigma)
        self.dispersion = lognorm.var(self.s, loc=self.mu, scale=self.sigma)
    
    def density(self, x):
        return lognorm.pdf(x, self.s, loc=self.mu, scale=self.sigma)    

    def func(self, x):
        return lognorm.cdf(x, self.s, loc=self.mu, scale=self.sigma)

    def gen_sample(self, n):
        return lognorm.rvs(self.s, loc=self.mu, scale=self.sigma, size=n)

class Rayleigh(Distribution):
    name = "Rayleigh"
    def __init__(self, mode=0, elem=None, sample=None):
        if mode == 0:
            self.mu = elem[0]
            self.sigma = elem[1]
        else:
            self.mu, self.sigma = rayleigh.fit(sample)
        self.math_average = rayleigh.mean(loc=self.mu, scale=self.sigma)
        self.dispersion = rayleigh.var(loc=self.mu, scale=self.sigma)
    
    def density(self, x):
        return rayleigh.pdf(x, loc=self.mu, scale=self.sigma)    

    def func(self, x):
        return rayleigh.cdf(x, loc=self.mu, scale=self.sigma)
    
    def gen_sample(self, n):
        return rayleigh.rvs(loc=self.mu, scale=self.sigma, size=n)

class Wald(Distribution):
    name = "Wald"
    def __init__(self, mode=0, elem=None, sample=None):
        if mode == 0:
            self.mu = elem[0]
            self.sigma = elem[1]
        else:
            self.mu, self.sigma = wald.fit(sample)
        self.math_average = wald.mean(loc=self.mu, scale=self.sigma)
        self.dispersion = wald.var(loc=self.mu, scale=self.sigma)
    
    def density(self, x, _mu=None, _sigma=None):
        return wald.pdf(x, loc=_mu, scale=_sigma)    

    def func(self, x, _mu=None, _sigma=None):
        return wald.cdf(x, loc=_mu, scale=_sigma)
    
    def gen_sample(self, n, _mu=None, _sigma=None):
        return wald.rvs(loc=_mu, scale=_sigma, size=n)

    def get_est_n_sample(self, sample):
        _mu, _sigma = wald.fit(sample)
        return _mu, _sigma 

class JohnsonSu(Distribution):
    name = "Johnson Su"
    def __init__(self, mode=0, elem=None, sample=None):
        if mode == 0:
            self.a = elem[0]
            self.b = elem[1]
            self.mu = elem[2]
            self.sigma = elem[3]
        else:
            self.a, self.b, self.mu, self.sigma = johnsonsu.fit(sample)
        self.math_average = johnsonsu.mean(self.a, self.b, loc=self.mu, scale=self.sigma)
        self.dispersion = johnsonsu.var(self.a, self.b, loc=self.mu, scale=self.sigma)
    
    def density(self, x):
        return johnsonsu.pdf(x, self.a, self.b, loc=self.mu, scale=self.sigma)    

    def func(self, x):
        return johnsonsu.cdf(x, self.a, self.b, loc=self.mu, scale=self.sigma)

    def gen_sample(self, n):
        return johnsonsu.rvs(self.a, self.b, loc=self.mu, scale=self.sigma, size=n)


class Rice(Distribution):
    name = "Rice"
    def __init__(self, mode=0, elem=None, sample=None):
        if mode == 0:
            self.b = elem[0]
            self.mu = elem[1]
            self.sigma = elem[2]
        else:
            self.b, self.mu, self.sigma = rice.fit(sample)
        self.math_average = rice.mean(self.b, loc=self.mu, scale=self.sigma)
        self.dispersion = rice.var(self.b, loc=self.mu, scale=self.sigma)
    
    def density(self, x):
        return rice.pdf(x, self.b, loc=self.mu, scale=self.sigma)    

    def func(self, x):
        return rice.cdf(x, self.b, loc=self.mu, scale=self.sigma)

    def gen_sample(self, n):
        return rice.rvs(self.b, loc=self.mu, scale=self.sigma, size=n)

class Change_Distribution(object):
    def __init__(self):
        self._distributions = []

    def add_distr(self, h):
        self._distributions.append(h)

    def response(self, sample, name=None):
        for h in self._distributions[0]:
            F = h.func(sample)
            print('Ответ: %s' % F[0])
