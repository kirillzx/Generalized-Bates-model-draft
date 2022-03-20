import numpy as np
import scipy.integrate as integrate
import scipy.stats as st
import scipy.special as sp
import enum
import scipy.optimize as optimize
from scipy.optimize import minimize

def ChFHestonModel(r, tau, kappa, gamma, vbar, v0, rho):
    i = complex(0.0, 1.0)
    D1 = lambda u: np.sqrt(np.power(kappa-gamma*rho*i*u, 2)+(u**2 + i*u) * gamma**2)
    g  = lambda u: (kappa-gamma*rho*i*u-D1(u))/(kappa-gamma*rho*i*u + D1(u))
    C  = lambda u: (1.0-np.exp(-D1(u)*tau))/(gamma*gamma*(1.0-g(u)*np.exp(-D1(u)*tau)))\
        *(kappa-gamma*rho*i*u-D1(u))
    # Note that we exclude the term -r*tau, as the discounting is performed in the COS method
    A  = lambda u: r * i*u *tau + kappa*vbar*tau/gamma/gamma *(kappa-gamma*rho*i*u-D1(u))\
        - 2*kappa*vbar/gamma/gamma * np.log((1.0-g(u)*np.exp(-D1(u)*tau))/(1.0-g(u)))

    cf = lambda u: np.exp(A(u) + C(u)*v0)
    return cf

def ChFBlackScholes(r, sigma, tau):
    cf = lambda u: np.exp((r - 0.5 * sigma**2)* 1j * u * tau - 0.5 * sigma**2 * u**2 * tau)
    return cf

def ChFBatesModel(r,tau,kappa,gamma,vbar,v0,rho,xiP,muJ,sigmaJ):
    i = complex(0.0,1.0)
    D1 = lambda u: np.sqrt(np.power(kappa-gamma*rho*i*u,2)+(u*u+i*u)*gamma*gamma)
    g  = lambda u: (kappa-gamma*rho*i*u-D1(u))/(kappa-gamma*rho*i*u+D1(u))
    C  = lambda u: (1.0-np.exp(-D1(u)*tau))/(gamma*gamma*(1.0-g(u)*\
                               np.exp(-D1(u)*tau)))*(kappa-gamma*rho*i*u-D1(u))
    # Note that we exclude the term -r*tau, as the discounting is performed in the COS method
    AHes= lambda u: r * i*u *tau + kappa*vbar*tau/gamma/gamma *(kappa-gamma*\
        rho*i*u-D1(u)) - 2*kappa*vbar/gamma/gamma*np.log((1.0-g(u)*np.exp(-D1(u)*tau))/(1.0-g(u)))

    A = lambda u: AHes(u) - xiP * i * u * tau *(np.exp(muJ+0.5*sigmaJ*sigmaJ) - 1.0) + \
            xiP * tau * (np.exp(i*u*muJ - 0.5 * sigmaJ * sigmaJ * u * u) - 1.0)

    cf = lambda u: np.exp(A(u) + C(u)*v0)
    return cf

def EFun(tau, u, e1, e2, k, gamma):
    eFun = ((k - e1) / gamma**2) * (1 - np.exp(-e1*tau))/(1 - e2 * np.exp(-e1*tau))
    return eFun

def CFun(tau, u, c1, c2, kr, gammar):
    cFun = ((kr - c1) / gammar**2) * (1 - np.exp(-c1*tau))/(1 - c2 * np.exp(-c1*tau))
    return cFun

def DFun(tau, u, k, gamma, d1, l1, v0, vb, krho, T):
    dFun = gamma * d1 * ( np.exp(-k*T)*(v0 - vb)/(krho + k) + vb/krho + np.exp(-k*T)*(vb - v0)/(krho + k - l1) - vb/(krho - l1) + \
        (- np.exp(-k*T)*(v0-vb)/(k + krho) - vb/krho + vb/(krho - l1) - np.exp(-k*T)*(vb-v0)/(k + krho - l1) ) )
    return dFun

def AFun(tau, u, muJ, sigmaJ, xip, k, vb, gamma, l1, sigmarho, d1, v0, krho, T):
    i = complex(0, 1)

    I1 = -i*u*(np.exp(muJ+0.5*sigmaJ**2) - 1)*tau*xip + xip*(np.exp(muJ*i*u - 0.5*sigmaJ**2 * u**2) - 1)*tau +\
        k*vb*(k-e1)/gamma**2 * (tau - np.exp(-l1*tau)/(-l1))

    z = np.linspace(0, tau, 100)

    # I2

    f = lambda z, u: DFun(z, u, k, gamma, d1, l1, v0, vb, krho, T)

    I3 = 0.5*sigmarho**2 * integrate.trapz((f(z, u))**2, z).reshape(u.size, 1)

def ChFBates_StochIR_StochCor(tau, T, k, gamma, vbar, kr, gammar, krho, v0):
    i = complex(0.0, 1.0)

    #define E function
    e1 = np.sqrt(k**2 + gamma**2 * (u**2 + i*u))
    e2 = (k - e1) / (k + e1)

    eFun = lambda u: EFun(tau, u, e1, e2, k, gamma)

    #define C function
    c1 = np.sqrt(kr**2 + gammar**2 * (u**2 + i*u))
    c2 = (kr - c1) / (kr + c1)

    cFun = lambda u: CFun(tau, u, c1, c2, kr, gammar)

    #define D function
    d1 = i*u * (k - e1)/gamma**2
    l1 = -np.log( (np.exp(-e1) - e2*np.exp(-e1))/(1 - e2*np.exp(-e1)) )
    dFun = lambda u: DFun(tau, u, k, gamma, d1, l1, v0, vb, krho, T)

    cf = lambda u: np.exp(cFun(u)*r0 + dFun(u)*rho0 + eFun(u)*v0)

    return 0
