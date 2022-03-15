import numpy as np

def optionPriceMCGeneral(type_option,S,K,T,r):
    # S is a vector of Monte Carlo samples at T
    result = np.zeros([len(K),1])
    if type_option == 'c' or type_option == 1:
        for (idx,k) in enumerate(K):
            result[idx] = np.exp(-r*T)*np.mean(np.maximum(S-k,0.0))
    elif type_option == 'p' or type_option == -1:
        for (idx,k) in enumerate(K):
            result[idx] = np.exp(-r*T)*np.mean(np.maximum(k-S,0.0))
    return result.T[0]

def hestonEuler(NumberPaths,N,T,r,s0,kappa,gamma,rho,vbar,v0):
    Z1 = np.random.normal(0.0,1.0,[NumberPaths,N])
    Z2 = np.random.normal(0.0,1.0,[NumberPaths,N])

    W1 = np.zeros([NumberPaths, N + 1])
    W2 = np.zeros([NumberPaths, N + 1])
    V = np.zeros([NumberPaths, N + 1])
    X = np.zeros([NumberPaths, N + 1])

    V[:,0]=v0
    X[:,0]=np.log(s0)

    time = np.zeros([N+1])

    dt = T / float(N)
    for i in range(0, N):
        Z2[:,i] = rho * Z1[:,i] + np.sqrt(1.0 - rho**2)*Z2[:,i]

        V[:,i+1] = V[:,i] + kappa*(vbar - V[:,i]) * dt + gamma * np.sqrt(V[:,i]) * np.sqrt(dt) * Z1[:,i]
        V[:,i+1] = np.maximum(V[:,i+1],0.0)

        X[:,i+1] = X[:,i] + (r - 0.5*V[:,i])*dt + np.sqrt(V[:,i]) * np.sqrt(dt) * Z2[:,i]
        time[i+1] = time[i] +dt

    S = np.exp(X)

    return time, S

def hestonMilstein(NumberPaths, N, T, r, s0, kappa, gamma, rho, vbar, v0):
    Z1 = np.random.normal(0.0,1.0,[NumberPaths,N])
    Z2 = np.random.normal(0.0,1.0,[NumberPaths,N])
    W1 = np.zeros([NumberPaths, N+1])
    W2 = np.zeros([NumberPaths, N+1])

    V = np.zeros([NumberPaths, N+1])
    X = np.zeros([NumberPaths, N+1])

    V[:,0]=v0
    X[:,0]=s0

    time = np.zeros([N+1])
    dt = T / float(N)

    for i in range(0,N):
        Z2[:,i] = rho * Z1[:,i] + np.sqrt(1.0 - rho**2)*Z2[:,i]

        V[:,i+1] = V[:,i] + kappa * (vbar - V[:,i]) * dt + gamma * np.sqrt(V[:,i]) * np.sqrt(dt) * Z1[:,i] +\
                                0.25 * gamma * ((W1[:,i+1]-W1[:,i])**2 - dt)
        V[:,i+1] = np.maximum(V[:,i+1],0.0)

        X[:, i+1] = X[:, i] + (r * X[:, i]) * dt + np.sqrt(V[:, i]) * X[:, i] * np.sqrt(dt) * Z2[:,i]+\
                                0.5 * V[:, i] * X[:, i] * ((np.sqrt(dt) * Z2[:,i])**2 - dt)
        time[i+1] = time[i] + dt

    S = X

    return time, S

def heston_stoch_corr(NumberPaths, N, T, r, v0, s0, rho0, rho1, rho2, kappa, gamma, rho, vbar, kappa_rho, mu_rho, sigma_rho):
    Z1 = np.random.normal(0.0,1.0, [NumberPaths,N])
    Z2 = np.random.normal(0.0,1.0, [NumberPaths,N])
    Z3 = np.random.normal(0.0, 1.0, [NumberPaths,N])

    W1 = np.zeros([NumberPaths, N + 1])
    W2 = np.zeros([NumberPaths, N + 1])
    W3 = np.zeros([NumberPaths, N + 1])

    V = np.zeros([NumberPaths, N+1])
    X = np.zeros([NumberPaths, N+1])
    rho = np.zeros([NumberPaths, N+1])

    V[:,0] = v0
    X[:,0] = np.log(s0)
    rho[:, 0] = rho0

    time = np.zeros([N+1])

    dt = T / float(N)
    for i in range(0, N):
        Z2[:, i] = rho1 * Z1[:, i] + np.sqrt(1 - rho1**2) * Z2[:, i]
        Z3[:, i] = rho[:, i] * Z1[:, i] + (rho2 - rho1 * rho[:, i])/np.sqrt(1 - rho1**2) * Z2[:, i] + \
                                        np.sqrt(abs(1 - rho[:, i]**2 - ((rho2 - rho1 * rho[:, i])/np.sqrt(1 - rho1**2))**2)) * Z3[:, i]


        W1[:, i+1] = W1[:,i] + np.sqrt(dt) * Z1[:,i]
        W2[:, i+1] = W2[:,i] + np.sqrt(dt) * Z2[:,i]
        W3[:, i+1] = W3[:,i] + np.sqrt(dt) * Z3[:,i]

        V[:, i+1] = V[:,i] + kappa*(vbar - V[:,i]) * dt + gamma* np.sqrt(V[:,i]) * (W1[:,i+1] - W1[:,i])
        V[:, i+1] = np.maximum(V[:,i+1], 0.0)

        rho[:, i+1] = rho[:, i] + kappa_rho * (mu_rho - rho[:, i]) * dt + sigma_rho * (W2[:,i+1] - W2[:,i])

        X[:,i+1] = X[:,i] + (r - 0.5 * V[:,i]) * dt + np.sqrt(V[:,i]) * (W3[:,i+1]-W3[:,i])
        time[i+1] = time[i] +dt

    S = np.exp(X)
    paths = {"time":time, "S":S, 'rho': rho}

    return paths

def CIR_exact(numberPaths, kappa, gamma, vbar, s, t, v_s):
    delta = 4.0 * kappa * vbar/gamma**2

    c = gamma**2/(4.0*kappa) * (1 - np.exp(-kappa * (t-s)))

    kappaBar = 4 * kappa * v_s * np.exp(-kappa * (t-s))/(gamma**2 * (1 - np.exp(-kappa * (t-s))))

    return  c * np.random.noncentral_chisquare(delta, kappaBar, numberPaths)

def hestonAES(numberPaths, N, s0, v0, T, kappa, gamma, vbar, rho, r):
    X = np.zeros([numberPaths, N + 1])
    S = np.zeros([numberPaths, N + 1])
    V = np.zeros([numberPaths, N + 1])
    time = np.zeros(N + 1)

    Zx = np.random.normal(0, 1, [numberPaths, N])

    X[:, 0] = np.log(s0)
    V[:, 0] = v0

    dt = T/float(N)

    for t in range(N):

        V[:, t+1] = CIR_exact(numberPaths, kappa, gamma, vbar, 0, dt, V[:, t])

        X[:, t+1] = X[:, t] + (r - vbar*kappa*rho/gamma) * dt + ((kappa*rho/gamma - 0.5) * dt - rho/gamma) * V[:, t] +\
                                        rho/gamma * V[:, t+1] + np.sqrt((1-rho**2) * dt * V[:, t]) * Zx[:, t]

        time[t+1] = time[t] + dt
    S = np.exp(X)

    return time, S, V

def heston_stoch_corr_AES(numberPaths, N, s0, v0, T, kappa, gamma, vbar, rho0, r, kp, mup, sigmap, rho1, rho2):
    X = np.zeros([numberPaths, N + 1])
    S = np.zeros([numberPaths, N + 1])
    V = np.zeros([numberPaths, N + 1])
    rho = np.zeros([numberPaths, N + 1])
    time = np.zeros(N + 1)

    Zx = np.random.normal(0, 1, [numberPaths, N])
    Zrho = np.random.normal(0, 1, [numberPaths, N])

    X[:, 0] = np.log(s0)
    V[:, 0] = v0
    rho[:, 0] = rho0

    dt = T/float(N)

    for t in range(N):

        V[:, t+1] = CIR_exact(numberPaths, kappa, gamma, vbar, 0, dt, V[:, t])
        rho[:, t+1] = rho[:, t]*np.exp(-kp*dt) + mup*(1 - np.exp(-kp*dt)) + sigmap*np.sqrt((1-np.exp(-2*kp*dt))/(2*kp))  * Zrho[:, t]

        X[:, t+1] = X[:, t] + (r - 0.5* V[:, t])*dt + rho[:, t]/gamma * (V[:, t+1] - V[:, t] - kappa*(vbar - V[:, t])*dt) + \
            (rho2-rho1*rho[:, t])/np.sqrt(1-rho1**2) * np.sqrt(V[:, t]) * 1/(np.sqrt(1-rho1**2)*sigmap) * (rho[:, t+1] - rho[:, t] - \
            kp*(mup - rho[:, t])*dt - rho1*sigmap* 1/(gamma*np.sqrt(V[:, t]))  * (V[:, t+1] - V[:, t] - kappa*(vbar - V[:, t])*dt) ) + \
            np.sqrt(1- (rho[:, t])**2 - ((rho2-rho1*rho[:, t])/np.sqrt(1-rho1**2))**2)*np.sqrt(V[:, t]) * np.sqrt(dt) * Zx[:, t]

        time[t+1] = time[t] + dt
    S = np.exp(X)

    return time, S, V