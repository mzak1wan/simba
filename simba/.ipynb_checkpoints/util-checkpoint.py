import pandas as pd
import numpy as np


def put_in_batch_form(data, name, verbose):
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)
        if verbose > 1:
            print(f'Assuming one batch only, reshaped {name} to {data.shape}')
    return data

def check_and_initialize_data(U, U_val=None, X=None, X_val=None, Y=None, Y_val=None, x0=None, x0_val=None, verbose=0, input_output=False):
    if U_val is None:
        U_val = U.copy()
        if verbose > 1:
            print('U_val was not provided --> set U_val=U')
    U = put_in_batch_form(U, 'U', verbose)
    U_val = put_in_batch_form(U_val, 'U_val', verbose)

    if input_output:
        assert Y is not None, 'Y must be provided for input output model fitting'
        if Y_val is None:
            Y_val = Y.copy()
            if verbose > 1:
                print('Y_val was not provided --> set Y_val=Y')
        Y = put_in_batch_form(Y, 'Y', verbose)
        Y_val = put_in_batch_form(Y_val, 'Y_val', verbose)

        assert U.shape[0] == Y.shape[0], f'U and Y must have the same first dimension, got {U.shape[0]} != {Y.shape[0]}'
        assert U_val.shape[0] == Y_val.shape[0], f'U_val and Y_val must have the same first dimension, got {U_val.shape[0]} != {Y_val.shape[0]}'
        assert U.shape[1] == Y.shape[1], f'U and Y must have the same number of samples, got {U.shape[1]} != {Y.shape[1]}'
        assert U_val.shape[1] == Y_val.shape[1], f'U_val and Y_val must have the same number of samples, got {U_val.shape[1]} != {Y_val.shape[1]}'
    else:
        assert X is not None, 'X must be provided for input state model fitting'
        if X_val is None:
            X_val = X.copy()
            if verbose > 1:
                print('X_val was not provided --> set X_val=X')
        X = put_in_batch_form(X, 'X', verbose)
        X_val = put_in_batch_form(X_val, 'X_val', verbose)
        
        assert U.shape[0] == X.shape[0], f'U and X must have the same first dimension, got {U.shape[0]} != {X.shape[0]}'
        assert U_val.shape[0] == X_val.shape[0], f'U_val and X_val must have the same first dimension, got {U_val.shape[0]} != {X_val.shape[0]}'
        assert U.shape[1] == X.shape[1], f'U and X must have the same number of samples, got {U.shape[1]} != {X.shape[1]}'
        assert U_val.shape[1] == X_val.shape[1], f'U_val and X_val must have the same number of samples, got {U_val.shape[1]} != {X_val.shape[1]}'
    
    if (x0 is not None) and (x0_val is None):
        x0_val = x0.copy()
    x0 = put_in_batch_form(x0, 'x0', verbose)
    x0_val = put_in_batch_form(x0_val, 'x0_val', verbose)
        
    return U, U_val, X, X_val, Y, Y_val, x0, x0_val
    
    
def generate_A_Hurwitz(nx):
    # https://math.stackexchange.com/questions/2674083/randomly-generate-hurwitz-matrices
    while True:
        try:
            W = np.diag(np.random.uniform(-1,1,(nx,)))
            V = np.random.normal(0, 1, (nx,nx))
            A = V.dot(W).dot(np.linalg.inv(V))
            break
        except:
            continue
    return A

def voss_noise(nrows, ncols=16):
    """Generates pink noise using the Voss-McCartney algorithm.
    
    nrows: number of values to generate
    rcols: number of random sources to add
    
    returns: NumPy array
    """
    array = np.empty((nrows, ncols))
    array.fill(np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(nrows)
    
    # the total number of changes is nrows
    n = nrows
    cols = np.random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=n)
    array[rows, cols] = np.random.random(n)

    df = pd.DataFrame(array)
    df.fillna(method='ffill', axis=0, inplace=True)
    total = df.sum(axis=1)

    return total.values
