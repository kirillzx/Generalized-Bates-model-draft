import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.integrate import quad
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d
from functools import partial
from scipy.optimize import minimize, fsolve, newton


def CallPutOptionPriceCOS(cf, type_option, s0, r, tau, K, N, L):
    # L    - size of truncation domain (typ.:L=8 or L=10)

    # reshape K to a column vector
    K = np.array(K).reshape([len(K),1])

    i = complex(0.0, 1.0)
    x0 = np.log(s0 / K)

    # truncation domain
    a = 0.0 - L * np.sqrt(tau)
    b = 0.0 + L * np.sqrt(tau)

    k = np.linspace(0,N-1,N).reshape([N,1])
    u = k * np.pi / (b - a);

    H_k = Hk_Coefficients(type_option,a,b,k)

    mat = np.exp(i * np.outer((x0 - a) , u))

    temp = cf(u) * H_k
    temp[0] = 0.5 * temp[0]

    value = np.exp(-r * tau) * K * np.real(mat.dot(temp))

    return value

def Hk_Coefficients(type_option, a, b, k):
    if type_option == "c" or type_option == 1:
        c = 0.0
        d = b
        coef = Chi_Psi(a, b, c, d, k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        if a < b and b < 0.0:
            H_k = np.zeros([len(k),1])
        else:
            H_k = 2.0 / (b - a) * (Chi_k - Psi_k)

    elif type_option == "p" or type_option == -1:
        c = a
        d = 0.0
        coef = Chi_Psi(a, b, c, d, k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        H_k = 2.0 / (b - a) * (- Chi_k + Psi_k)

    return H_k

def Chi_Psi(a, b, c, d, k):
    psi = np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a)/(b - a))
    psi[1:] = psi[1:] * (b - a) / (k[1:] * np.pi)
    psi[0] = d - c

    chi = 1.0 / (1.0 + np.power((k * np.pi / (b - a)) , 2.0))
    expr1 = np.cos(k * np.pi * (d - a)/(b - a)) * np.exp(d)  - np.cos(k * np.pi
                  * (c - a) / (b - a)) * np.exp(c)
    expr2 = k * np.pi / (b - a) * np.sin(k * np.pi *
                        (d - a) / (b - a))   - k * np.pi / (b - a) * np.sin(k
                        * np.pi * (c - a) / (b - a)) * np.exp(c)
    chi = chi * (expr1 + expr2)

    value = {"chi":chi,"psi":psi }
    return value

def BS_Call_Option_Price(type_option, s0, K, sigma, tau, r):
    # if isinstance(K, list):
    #     K = np.array(K).reshape([len(K),1])
    K = np.array(K)

    d1 = (np.log(s0 / K) + (r + 0.5 * np.power(sigma,2.0))* tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    if type_option == 'c' or type_option == 1:
        value = stats.norm.cdf(d1) * s0 - stats.norm.cdf(d2) * K * np.exp(-r * tau)
    elif type_option == 'p' or type_option == -1:
        value = stats.norm.cdf(-d2) * K * np.exp(-r * tau) - stats.norm.cdf(-d1)*s0

    return value


def optionPriceCOSMthd_StochIR(cf, CP, s0,tau,K,N,L,P0T):

    # cf   - Characteristic function is a function, in the book denoted by \varphi
    # CP   - C for call and P for put
    # s0   - Initial stock price
    # tau  - Time to maturity
    # K    - List of strikes
    # N    - Number of expansion terms
    # L    - Size of truncation domain (typ.:L=8 or L=10)
    # P0T  - Zero-coupon bond for maturity T.

    # Reshape K to become a column vector

    if K is not np.array:
        K = np.array(K).reshape([len(K),1])

    # Assigning i=sqrt(-1)

    i = np.complex(0.0,1.0)
    x0 = np.log(s0 / K)

    # Truncation domain

    a = 0.0 - L * np.sqrt(tau)
    b = 0.0 + L * np.sqrt(tau)

    # Summation from k = 0 to k=N-1

    k = np.linspace(0,N-1,N).reshape([N,1])
    u = k * np.pi / (b - a)

    # Determine coefficients for put prices

    H_k = CallPutCoefficients(OptionType.PUT,a,b,k)
    mat = np.exp(i * np.outer((x0 - a) , u))
    temp = cf(u) * H_k
    temp[0] = 0.5 * temp[0]
    value = K * np.real(mat.dot(temp))

    # We use the put-call parity for call options

    if CP == OptionType.CALL:
        value = value + s0 - K * P0T

    return value
