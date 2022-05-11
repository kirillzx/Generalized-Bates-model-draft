import numpy as np
import scipy.integrate as integrate
import scipy.stats as st
import scipy.special as sp
import enum
import scipy.optimize as optimize
from scipy.optimize import minimize
import cmath

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

def EFun(tau, u, k, gamma):
    i = complex(0, 1)
    e1 = np.sqrt(k**2 + (gamma**2) * (u**2 + i*u) + 0*i)
    e2 = (k - e1) / (k + e1)

    eFun = ((k - e1) / gamma**2) * (1 - np.exp(-e1*tau))/(1 - e2 * np.exp(-e1*tau))
    return eFun

def CFun(tau, u, kr, gammar):
    i = complex(0, 1)
    c1 = np.sqrt(- (kr)**2 - 2 * gammar**2 + 2*i*u* gammar**2 + 0*i)

    cFun = (kr + c1*np.tan(0.5 * c1 * tau - np.arctan(kr/c1)))/(gammar**2)
    return cFun

def DFun(tau, u, k, gamma, v0, vb, krho, T):
    i = complex(0, 1)
    e1 = np.sqrt(k**2 + (gamma**2) * (u**2 + i*u) + 0*i)
    e2 = (k - e1) / (k + e1)
    d1 = i*u * (k - e1)/(gamma**2)
    l1 = -np.log((np.exp(-e1) - e2*np.exp(-e1))/(1 - e2*np.exp(-e1)))

    dFun = gamma * d1 * ( np.exp(-k*T + k*tau)*(v0 - vb)/(krho + k) + vb/krho + np.exp(-k*T+k*tau-l1*tau)*(vb - v0)/(krho + k - l1) - np.exp(-l1*tau)*vb/(krho - l1) + \
        (- np.exp(-k*T)*(v0-vb)/(k + krho) - vb/krho + vb/(krho - l1) - np.exp(-k*T)*(vb-v0)/(k + krho - l1) )*np.exp(-krho*tau) )
    return dFun


def AFun(tau, u, muJ, sigmaJ, xip, k, vb, gamma, v0, krho, murho,
                        sigmarho, rho4, rho5, T, kr, gammar, mur, r0):
    i = complex(0, 1)

    e1 = np.sqrt(k**2 + (gamma**2) * (u**2 + i*u) + 0*i)
    e2 = (k - e1) / (k + e1)
    d1 = i*u * (k - e1)/gamma**2
    l1 = -np.log( (np.exp(-e1) - e2*np.exp(-e1))/(1 - e2*np.exp(-e1)) )
    c1 = np.sqrt(- (kr)**2 - 2 * gammar**2 + 2*i*u* gammar**2 + 0*i)

    # c1 = np.sqrt(kr**2 + gammar**2 * (u**2 + i*u))
    # c2 = (kr - c1) / (kr + c1)

    ct = lambda t: 1/(4*k) * gamma**2 * (1 - np.exp(-k*t))
    d11 = 4*k*vb/(gamma**2)
    lambda1t = lambda t: (4*k*v0*np.exp(-k*t))/(gamma**2 * (1 - np.exp(-k*t)))

    L1 = lambda t: np.sqrt(ct(t) * (lambda1t(t) - 1) + ct(t)*d11 + (ct(t)*d11)/(2*(d11+lambda1t(t))) + 0*i)

    c2t = lambda t: 1/(4*kr) * gammar**2 * (1 - np.exp(-kr*t))
    d2 = 4*kr*mur/(gammar**2)
    lambda2t = lambda t: (4*kr*r0*np.exp(-kr*t))/(gammar**2 * (1 - np.exp(-kr*t)))

    L2 = lambda t: np.sqrt(c2t(t) * (lambda2t(t) - 1) + c2t(t)*d2 + (c2t(t)*d2)/(2*(d2+lambda2t(t))) + 0*i)

    a = np.sqrt(vb - (gamma**2)/(8*k) + 0*i)
    b = np.sqrt(v0 + 0*i) - a
    c = - np.log(1/b * (L1(1) - a) + 0*i)
    m = np.sqrt(mur - (gammar**2)/(8*kr) + 0*i)
    n = np.sqrt(r0 + 0*i) - m
    o = - np.log(1/n * (L2(1) - m) + 0*i)

    I1 = -i*u*(np.exp(muJ+0.5*sigmaJ**2) - 1)*tau*xip + xip*(np.exp(muJ*i*u - 0.5*sigmaJ**2 * u**2) - 1)*tau +\
        k*vb*(k-e1)/(gamma**2) * (tau - np.exp(-l1*tau)/(-l1))

    z = np.linspace(0, tau, 500)

    # f_I21 = lambda z1,u: np.exp(-c*(T-z1)) * DFun(z1, u, k, gamma, d1, l1, v0, vb, krho, T)
    # I21 = [integrate.trapz(np.array(list(map(lambda z1: f_I21(z1, var), z))), z) for var in u]

    # f_I21 = np.exp(-c*(T-z)) * DFun(z, u, k, gamma, d1, l1, v0, vb, krho, T)

    f_I21 = lambda z, u: DFun(z, u, k, gamma, v0, vb, krho, T)
    I21 = integrate.trapz(f_I21(z, u), z).reshape(u.size,1)

    f_I22 = lambda z, u: np.exp(-c*(T-z)) * DFun(z, u, k, gamma, v0, vb, krho, T)
    I22 = integrate.trapz(f_I22(z, u), z).reshape(u.size,1)

    f_I23 = lambda z, u: (DFun(z, u, k, gamma, v0, vb, krho, T))**2
    I23 = integrate.trapz(f_I23(z, u), z).reshape(u.size,1)

    I2 = (krho*murho + sigmarho*rho4*i*u*a)*I21 + sigmarho*rho4*i*u*b * I22 + (0.5*sigmarho**2) * I23


    # f_I41 = lambda z,u: CFun(z, u, c1, c2, kr, gammar)
    # f_I42 = lambda z,u: np.exp(-o*(T-z)) * CFun(z, u, c1, c2, kr, gammar)
    # f_I43 = lambda z,u: np.exp(-c*(T-z)) * CFun(z, u, c1, c2, kr, gammar)
    # f_I44 = lambda z,u: np.exp((-o-c)*(T-z)) * CFun(z, u, c1, c2, kr, gammar)
    f_I31 = lambda z, u: CFun(z, u, kr, gammar)
    f_I32 = lambda z, u: np.exp(-c*(T-z)) * CFun(z, u, kr, gammar)
    f_I33 = lambda z, u: np.exp(-o*(T-z)) * CFun(z, u, kr, gammar)
    f_I34 = lambda z, u: np.exp((-o-c)*(T-z)) * CFun(z, u, kr, gammar)

    # I41 = [integrate.trapz(np.array(list(map(lambda z: f_I41(z, var), z))), z) for var in u]
    # I42 = [integrate.trapz(np.array(list(map(lambda z: f_I42(z, var), z))), z) for var in u]
    # I43 = [integrate.trapz(np.array(list(map(lambda z: f_I43(z, var), z))), z) for var in u]
    # I44 = [integrate.trapz(np.array(list(map(lambda z: f_I44(z, var), z))), z) for var in u]
    I31 = integrate.trapz(f_I31(z, u), z).reshape(u.size,1)
    I32 = integrate.trapz(f_I32(z, u), z).reshape(u.size,1)
    I33 = integrate.trapz(f_I33(z, u), z).reshape(u.size,1)
    I34 = integrate.trapz(f_I34(z, u), z).reshape(u.size,1)

    I3 = i*u*gammar*rho5 * (m*a*I31 + m*b*I32 + n*a*I33 + n*b*I34)

    return I1 + I2  + I3

def ChFBates_StochIR_StochCor(tau, T, k, gamma, vb, kr, gammar, mur, krho, murho, sigmarho, rho4, rho5,
                    xip, muJ, sigmaJ, v0, r0, rho0):
    i = complex(0.0, 1.0)

    #define E function
    # e1 = lambda u: np.sqrt(k**2 + gamma**2 * (u**2 + i*u))
    # e2 = lambda u: (k - e1(u)) / (k + e1(u))
    #
    eFun = lambda u: EFun(tau, u, k, gamma)
    #
    # #define C function
    # c1 = lambda u: np.sqrt(kr**2 + gammar**2 * (u**2 + i*u))
    # c2 = lambda u: (kr - c1(u)) / (kr + c1(u))
    #
    cFun = lambda u: CFun(tau, u, kr, gammar)
    # #define D function
    # d1 = lambda u: i*u * (k - e1(u))/gamma**2
    # l1 = lambda u: -np.log( (np.exp(-e1(u)) - e2(u)*np.exp(-e1(u)))/(1 - e2(u)*np.exp(-e1(u))) )

    dFun = lambda u: DFun(tau, u, k, gamma, v0, vb, krho, T)

    #define A function
    aFun = lambda u: AFun(tau, u, muJ, sigmaJ, xip, k, vb, gamma, v0, krho, murho,
                            sigmarho, rho4, rho5, T, kr, gammar, mur, r0)

    cf = lambda u: np.exp(aFun(u) + cFun(u)*r0 + dFun(u)*rho0 + eFun(u)*v0)

    return cf

def AFun_DCL(tau, u, muJ, sigmaJ, xip, k, vb, gamma, v0, krho, delta,
                        rho4, rho5, T, kr, gammar, mur, r0, rho0):
    i = complex(0, 1)

    e1 = np.sqrt(k**2 + (gamma**2) * (u**2 + i*u) + 0*i)
    e2 = (k - e1) / (k + e1)
    d1 = i*u * (k - e1)/gamma**2
    l1 = -np.log( (np.exp(-e1) - e2*np.exp(-e1))/(1 - e2*np.exp(-e1)) )
    c1 = np.sqrt(- (kr)**2 - 2 * gammar**2 + 2*i*u* gammar**2 + 0*i)
    # c1 = np.sqrt(kr**2 + gammar**2 * (u**2 + i*u))
    # c2 = (kr - c1) / (kr + c1)

    ct = lambda t: 1/(4*k) * gamma**2 * (1 - np.exp(-k*t))
    d11 = 4*k*vb/(gamma**2)
    lambda1t = lambda t: (4*k*v0*np.exp(-k*t))/(gamma**2 * (1 - np.exp(-k*t)))

    L1 = lambda t: np.sqrt(ct(t) * (lambda1t(t) - 1) + ct(t)*d11 + (ct(t)*d11)/(2*(d11+lambda1t(t))) + 0*i)

    c2t = lambda t: 1/(4*kr) * gammar**2 * (1 - np.exp(-kr*t))
    d2 = 4*kr*mur/(gammar**2)
    lambda2t = lambda t: (4*kr*r0*np.exp(-kr*t))/(gammar**2 * (1 - np.exp(-kr*t)))

    L2 = lambda t: np.sqrt(c2t(t) * (lambda2t(t) - 1) + c2t(t)*d2 + (c2t(t)*d2)/(2*(d2+lambda2t(t))) + 0*i)

    a = np.sqrt(vb - (gamma**2)/(8*k) + 0*i)
    b = np.sqrt(v0 + 0*i) - a
    c = - np.log(1/b * (L1(1) - a) + 0*i)
    m = np.sqrt(mur - (gammar**2)/(8*kr) + 0*i)
    n = np.sqrt(r0 + 0*i) - m
    o = - np.log(1/n * (L2(1) - m) + 0*i)

    L3 = lambda t: np.sqrt(1 - (1/(2*(delta+1)+1) + (rho0 - 1/(2*(delta+1)+1))*np.exp(-2/krho - 1/(krho*(delta+1))*t) - (np.exp(-1/krho * t)*rho0)**4)/(1 - (np.exp(-1/krho * t)*rho0)**2) + 0*i)
    crho = np.sqrt(1 - 1/(2*(delta+1) + 1) + 0*i)
    arho = np.sqrt(1 - (1/(2*(delta+1)+1) + (rho0 - 1/(2*(delta+1) +1)) - rho0**4)/(1-rho0**2) + 0*i) -\
        np.sqrt(1 - 1/(2*(delta+1)+1) + 0*i)
    brho = -np.log((L3(1) - crho)/arho + 0*i)

    I1 = -i*u*(np.exp(muJ+0.5*sigmaJ**2) - 1)*tau*xip + xip*(np.exp(muJ*i*u - 0.5*sigmaJ**2 * u**2) - 1)*tau +\
        k*vb*(k-e1)/(gamma**2) * (tau - np.exp(-l1*tau)/(-l1))

    z = np.linspace(0, tau, 500)

    f_I21 = lambda z, u: DFun(z, u, k, gamma, v0, vb, 1/krho, T)
    I21 = integrate.trapz(f_I21(z, u), z).reshape(u.size,1)

    f_I22 = lambda z, u: np.exp(-(1/(krho*(delta+1)))*(T-z)) * (DFun(z, u, k, gamma, v0, vb, 1/krho, T))**2
    I22 = integrate.trapz(f_I22(z, u), z).reshape(u.size,1)

    f_I23 = lambda z, u: (DFun(z, u, k, gamma, v0, vb, 1/krho, T))**2
    I23 = integrate.trapz(f_I23(z, u), z).reshape(u.size,1)

    f_I24 = lambda z, u: np.exp(-brho*(T-z)) * DFun(z, u, k, gamma, v0, vb, 1/krho, T)
    I24 = integrate.trapz(f_I24(z, u), z).reshape(u.size,1)

    f_I25 = lambda z, u: np.exp((-c-brho)*(T-z)) * DFun(z, u, k, gamma, v0, vb, 1/krho, T)
    I25 = integrate.trapz(f_I25(z, u), z).reshape(u.size,1)

    f_I26 = lambda z, u: np.exp(-c*(T-z)) * DFun(z, u, k, gamma, v0, vb, 1/krho, T)
    I26 = integrate.trapz(f_I26(z, u), z).reshape(u.size,1)

    I2 = 1/(2*krho*(delta+1)) * (I23*(1 - 1/(2*(delta+1)+1)) + I22*(np.exp(-2/krho) * (1/(2*delta+1)+1) - rho0) +\
        rho4*i*u/np.sqrt(krho*(delta+1)) * (a*arho*I24 + a*crho*I21 + b*arho*I25 + b*crho*I26) )

    f_I31 = lambda z, u: CFun(z, u, kr, gammar)
    f_I32 = lambda z, u: np.exp(-c*(T-z)) * CFun(z, u, kr, gammar)
    f_I33 = lambda z, u: np.exp(-o*(T-z)) * CFun(z, u, kr, gammar)
    f_I34 = lambda z, u: np.exp((-o-c)*(T-z)) * CFun(z, u, kr, gammar)

    I31 = integrate.trapz(f_I31(z, u), z).reshape(u.size,1)
    I32 = integrate.trapz(f_I32(z, u), z).reshape(u.size,1)
    I33 = integrate.trapz(f_I33(z, u), z).reshape(u.size,1)
    I34 = integrate.trapz(f_I34(z, u), z).reshape(u.size,1)

    I3 = i*u*gammar*rho5 * (m*a*I31 + m*b*I32 + n*a*I33 + n*b*I34)

    return I1 + I2  + I3

def ChFBates_StochIR_StochCor_DCL(tau, T, k, gamma, vb, kr, gammar, mur, krho, delta, rho4, rho5,
                    xip, muJ, sigmaJ, v0, r0, rho0):
    i = complex(0.0, 1.0)
    eFun = lambda u: EFun(tau, u, k, gamma)
    cFun = lambda u: CFun(tau, u, kr, gammar)
    dFun = lambda u: DFun(tau, u, k, gamma, v0, vb, 1/krho, T)
    aFun = lambda u: AFun_DCL(tau, u, muJ, sigmaJ, xip, k, vb, gamma, v0, krho, delta,
                            rho4, rho5, T, kr, gammar, mur, r0, rho0)

    cf = lambda u: np.exp(aFun(u) + cFun(u)*r0 + dFun(u)*rho0 + eFun(u)*v0)

    return cf
