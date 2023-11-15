import numpy as np
import scipy

from slode.util import generate_A_Hurwitz, voss_noise


def generate_random_system(nx, nu, ny, N, stable_A=True):
    
    if stable_A:
        A = generate_A_Hurwitz(nx)
    else:
        A = np.random.normal(0, 1, (nx,nx))
    
    B = np.random.normal(0, 1, (nx,nu))
    C = np.random.normal(0, 1, (ny,nx))
    D = np.random.normal(0, 1, (ny,nu))
    
    return A, B, C, D
    
    
def generate_data(A, B, C, D, N, id_D, U=None, x0=None, gaussian_U=False, low_limit=-1, high_limit=1, mean=0, scale=1):
    
    if not id_D:
        lti = scipy.signal.dlti(A, B, C, np.zeros(D.shape), dt=1)
    else:
        lti = scipy.signal.dlti(A, B, C, D, dt=1)
    
    t = np.arange(N)
    if U is None:
        if gaussian_U:
            U = np.random.normal(mean, scale, (N, B.shape[1]))
        else:
            U = np.random.uniform(low_limit, high_limit, (N, B.shape[1]))
    if x0 is None:
        x0 = np.random.random((1, A.shape[1]))
        
    _, Y, X = lti.output(U, t, x0)
    
    return U, Y, X
    

def add_noise(*args, voss=False, colored=False, scale=1, ncols=16):
    for arg in args:
        if voss:
            noise = np.empty(arg.shape)
            for i in range(arg.shape[1]):
                noise[:,i] = voss_noise(arg.shape[0], ncols=ncols)
        elif colored:
            A, B, C, D = generate_random_system(nx=arg.shape[1], nu=arg.shape[1], ny=arg.shape[1], N=arg.shape[0], stable_A=True)
            _, noise, _ = generate_data(A, B, C, D, N=arg.shape[0], U=None, x0=None, gaussian_U=True)
        else:
            noise = np.random.random(arg.shape)
        arg = arg + noise * scale
    return args if len(args) > 1 else args[0]
