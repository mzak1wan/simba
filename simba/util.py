import pandas as pd
import numpy as np
import os, sys
import torch
import random
import warnings
import scipy
from copy import deepcopy

from contextlib import contextmanager
from timeit import default_timer

def format_elapsed_time(diff):
    """
    Small function to print the time elapsed between tic and toc in a nice manner
    """

    hours = int(diff // 3600)
    minutes = int((diff - 3600 * hours) // 60)
    seconds = str(int(diff - 3600 * hours - 60 * minutes))

    # Put everything in strings
    hours = str(hours)
    minutes = str(minutes)

    # Add a zero for one digit numbers for consistency
    if len(seconds) == 1:
        seconds = '0' + seconds
    if minutes == '0':
        return f"{seconds}\""

    if len(minutes) == 1:
        minutes = '0' + minutes
    if hours == '0':
        return f"{minutes}'{seconds}\""
    
    if len(hours) == 1:
        hours = '0' + hours
    return f"{hours}:{minutes}'{seconds}\""

def put_in_batch_form(data, name, verbose):
    if data is not None:
        if len(data.shape) == 2:
            if isinstance(data, np.ndarray):
                data = np.expand_dims(data, axis=0)
            else:
                data = data.unsqueeze(0)
            if verbose > 1:
                print(f'Assuming one batch only, reshaped {name} to {data.shape}')
    return data

def make_tensors(U, X=None, Y=None, x0=None, device=None):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = "cpu" 
    if not isinstance(U, torch.DoubleTensor):
        U = torch.tensor(U, dtype=torch.float64).to(device) if U is not None else None
        X = torch.tensor(X, dtype=torch.float64).to(device) if X is not None else None
        Y = torch.tensor(Y, dtype=torch.float64).to(device) if Y is not None else None
        x0 = torch.tensor(x0, dtype=torch.float64).to(device) if x0 is not None else None
    else:
        U = U.to(device) if U is not None else None
        X = X.to(device) if X is not None else None
        Y = Y.to(device) if Y is not None else None
        x0 = x0.to(device) if x0 is not None else None
    return U, X, Y, x0

def check_and_initialize_data(U=None, U_val=None, U_test=None, X=None, X_val=None, X_test=None, 
                              Y=None, Y_val=None, Y_test=None, x0=None, x0_val=None, x0_test=None, 
                              verbose=0, autonomous=False, input_output=False, device='cpu'):
    
    if not autonomous:
        assert U is not None, 'U has to be provided for non-autonomous systems.'
        if U_val is None:
            U_val = U.copy()
            if verbose > 1:
                print('U_val was not provided --> set U_val=U.')
        U = put_in_batch_form(U, 'U', verbose)
        U_val = put_in_batch_form(U_val, 'U_val', verbose)
        U_test = put_in_batch_form(U_test, 'U_val', verbose)

    if input_output:
        assert Y is not None, 'Y must be provided for input output model fitting.'
        if Y_val is None:
            Y_val = Y.copy()
            if verbose > 1:
                print('Y_val was not provided --> set Y_val=Y.')
        Y = put_in_batch_form(Y, 'Y', verbose)
        Y_val = put_in_batch_form(Y_val, 'Y_val', verbose)
        Y_test = put_in_batch_form(Y_test, 'Y_test', verbose)

        if U is None:
            # Trick to pass the batch size and prediction horizon in the forward pass:
            # U is actually never used, but u.shape[0] and u.shape[1] define the shape of the
            # predictions of Simba in the forward pass
            U = np.empty_like(Y, dtype='float64')
            U_val = np.empty_like(Y_val, dtype='float64')
        else:
            assert U.shape[0] == Y.shape[0], f'U and Y must have the same first dimension, got {U.shape[0]} != {Y.shape[0]}.'
            assert U_val.shape[0] == Y_val.shape[0], f'U_val and Y_val must have the same first dimension, got {U_val.shape[0]} != {Y_val.shape[0]}.'
            assert U.shape[1] == Y.shape[1], f'U and Y must have the same number of samples, got {U.shape[1]} != {Y.shape[1]}.'
            assert U_val.shape[1] == Y_val.shape[1], f'U_val and Y_val must have the same number of samples, got {U_val.shape[1]} != {Y_val.shape[1]}.'
    else:
        assert X is not None, 'X must be provided for input state model fitting.'
        if X_val is None:
            X_val = X.copy()
            if verbose > 1:
                print('X_val was not provided --> set X_val=X.')
        X = put_in_batch_form(X, 'X', verbose)
        X_val = put_in_batch_form(X_val, 'X_val', verbose)
        X_test = put_in_batch_form(X_test, 'X_test', verbose)
        
        if U is None:
            # Trick to pass the batch size and prediction horizon in the forward pass:
            # U is actually never used, but u.shape[0] and u.shape[1] define the shape of the
            # predictions of Simba in the forward pass
            U = np.empty_like(Y, dtype='float64')
            U_val = np.empty_like(Y_val, dtype='float64')
        else:
            assert U.shape[0] == X.shape[0], f'U and X must have the same first dimension, got {U.shape[0]} != {X.shape[0]}.'
            assert U_val.shape[0] == X_val.shape[0], f'U_val and X_val must have the same first dimension, got {U_val.shape[0]} != {X_val.shape[0]}.'
            assert U.shape[1] == X.shape[1], f'U and X must have the same number of samples, got {U.shape[1]} != {X.shape[1]}.'
            assert U_val.shape[1] == X_val.shape[1], f'U_val and X_val must have the same number of samples, got {U_val.shape[1]} != {X_val.shape[1]}.'    
        
    if (x0 is not None) and (x0_val is None):
        x0_val = deepcopy(x0)
    if (x0 is not None) and (x0_test is None):
        x0_test = deepcopy(x0_val)
    if x0 is not None:
        x0 = put_in_batch_form(x0, 'x0', verbose)
        x0_val = put_in_batch_form(x0_val, 'x0_val', verbose)
        x0_test = put_in_batch_form(x0_test, 'x0_test', verbose)

    if not isinstance(U, torch.DoubleTensor) or device is None:
        U, X, Y, x0 = make_tensors(U, X, Y, x0, device)
        U_val, X_val, Y_val, x0_val = make_tensors(U_val, X_val, Y_val, x0_val, device)
        U_test, X_test, Y_test, x0_test = make_tensors(U_test, X_test, Y_test, x0_test, device)
    
    if not autonomous:
        assert not torch.isnan(U).any(), f'U contains {torch.isnan(U).sum()} NaN(s), which is not handled currently.'
        assert not torch.isnan(U_val).any(), f'U_val contains {torch.isnan(U_val).sum()} NaN(s), which is not handled currently.'
        
    return U, U_val, U_test, X, X_val, X_test, Y, Y_val, Y_test, x0, x0_val, x0_test
    
def break_trajectories(data, horizon, stride=1):
    """
    Breaks trajectories in overlapping bits:
        Transforms data = (n_traj, N, :) 
        into data = (n_traj * ?, horizon+1, :)
    where '?' is determined by the desired overlapping between the created bits, N, and the horizon.
    """
    data = put_in_batch_form(data, name='', verbose=0)
    assert horizon < data.shape[1], f'Trajectories are not long enough for the desired horizon of {horizon}: at least {horizon + 1} points are required by trajectory, but {data.shape[1]} are provided.'
    n_traj = data.shape[0]
    N = data.shape[1]
    if isinstance(data, np.ndarray):
        out = np.concatenate([
            np.concatenate([
                data[[traj], n:n+horizon+1, :] for n in range(0, N-horizon, stride)
            ], axis=0) for traj in range(n_traj)], axis=0)
    else:
        out = torch.cat([
            torch.cat([
                data[[traj], n:n+horizon+1, :] for n in range(0, N-horizon, stride)
            ], axis=0) for traj in range(n_traj)], axis=0)
    return out

def evaluate(A, B, U, X, noise=None, name='', print_trajs=True, return_mean=False):
    H = X.shape[1]
    if not return_mean:
        print(f'{name}:', end='')
    errors = []
    maes = []
    mapes = []
    fros = []
    for j in range(U.shape[0]):
        x_all = torch.empty((X.shape[1], X.shape[2]), dtype=torch.float64, device=U.device)
        if noise is None:
            x = X[j,[0],:].T
        else:
            x = X[j,[0],:].T - noise[:,[j*H]]
        for i in range(X.shape[1]):
            x_all[[i],:] = x.T
            x = A @ x + B @ U[j,[i],:].T
        if noise is None:
            X_ = X[j,:,:]
        else:
            X_ = X[j,:,:] - noise[:,j*H:(j+1)*H].T
        errors.append(torch.mean((x_all - X_)**2).cpu().detach().numpy())
        maes.append(torch.mean(torch.abs(x_all - X_)).cpu().detach().numpy()) 
        mapes.append(torch.mean(torch.abs((x_all - X_)/X_)).cpu().detach().numpy()) 
        fros.append(torch.linalg.norm(x_all - X_, ord='fro').cpu().detach().numpy()**2/2)
        if print_trajs:
            print(f'\tTr. {j+1}:\t{errors[-1]:.2E}\t{maes[-1]:.2E}\t{mapes[-1]*100:.1f}%\t\t{fros[-1]:.2E}')
    if print_trajs:
        print('\t\t--------------------------------------------------------')
    if return_mean:
        return np.mean(errors)
    else:
        print(f'\t{"Mean:" if print_trajs else""}\t{np.mean(errors):.2E}\t{np.mean(maes):.2E}\t{np.mean(mapes)*100:.1f}%\t\t{np.mean(fros):.2E}')
    if print_trajs:
        print('')

def print_LS_SIMBa(A_ls, B_ls, A_simba, B_simba, U, X, noise, U_val, X_val, U_test, X_test, print_trajs=True):
    print('\nTraining\tMSE\t\tMAE\t\tMAPE\t\tFrob.\n')
    evaluate(A_ls, B_ls, U, X, None, 'LS', print_trajs)
    evaluate(A_simba, B_simba, U, X, None, 'SIMBa', print_trajs)
    print('________________________________________________________________________\n')

    if noise is not None:
        print('Train no noise\tMSE\t\tMAE\t\tMAPE\t\tFrob.\n')
        evaluate(A_ls, B_ls, U, X, noise, 'LS', print_trajs)
        evaluate(A_simba, B_simba, U, X, noise, 'SIMBa', print_trajs)
        print('________________________________________________________________________\n')

    print('Validation\tMSE\t\tMAE\t\tMAPE\t\tFrob.\n')
    evaluate(A_ls, B_ls, U_val, X_val, None, 'LS', print_trajs)
    evaluate(A_simba, B_simba, U_val, X_val, None, 'SIMBa', print_trajs)
    print('________________________________________________________________________\n')

    print('Test\t\tMSE\t\tMAE\t\tMAPE\t\tFrob.\n')
    evaluate(A_ls, B_ls, U_test, X_test, None, 'LS', print_trajs)
    evaluate(A_simba, B_simba, U_test, X_test, None, 'SIMBa', print_trajs)
    
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


def normalize(data, min_=None, diff=None):
    if min_ is None:
        min_ = data.min(axis=0, keepdim=True).values.min(axis=1, keepdim=True).values
        diff = data.max(axis=0, keepdim=True).values.max(axis=1, keepdim=True).values - min_
        return (data - min_) / diff * 0.8 + 0.1, min_, diff
    else:
        return (data - min_) / diff * 0.8 + 0.1

def inverse_normalize(data, min_, diff):
    return (data - 0.1) / 0.8 * diff + min_

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_results(directory, save_name, names, times, train_ids, validation_ids, test_ids, data, sim_params=None, data_params=None):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    savename = os.path.join(directory, save_name)
    torch.save(
        {
            'names': names,
            'times': times,
            'train_ids': train_ids,
            'validation_ids': validation_ids,
            'test_ids': test_ids,
            'sim_params': sim_params,
            'data_params': data_params, 
            'data': data,
        },
        savename,
    )

def load_results(directory, save_name):
    # Load the checkpoint
    checkpoint = torch.load(os.path.join(directory, save_name), map_location=lambda storage, loc: storage)
    names = checkpoint['names']
    times = checkpoint['times']
    train_ids = checkpoint['train_ids']
    validation_ids = checkpoint['validation_ids']
    test_ids = checkpoint['test_ids']
    data = checkpoint['data']
    sim_params = checkpoint['sim_params']
    data_params = checkpoint['data_params']

    return names, times, train_ids, validation_ids, test_ids, data, sim_params, data_params

def _squeeze(x):
     if x is not None:
        if x.shape[0] == 1:
            return x.squeeze(0)
        else:
            return x
     else:
        return x

def torch_to_mat(load_directory, load_name, save_directory, save_name):
    _, _, _, _, data, _, _ = load_results(load_directory, load_name)
    try:
        U, U_val, X, X_val, Y, Y_val, x0, x0_val = data
    except ValueError:
        if 'Franka' in load_directory:
            nu = 7
            nx = 17
            ny = 17
            H = 399
            U = data[:nu,:-H].T.reshape(-1, H, nu)
            U_val = data[:nu,-H:].T.reshape(-1, H, nu)
            X = data[nu:-ny,:-H].T.reshape(-1, H, nx)
            X_val = data[nu:-ny,-H:].T.reshape(-1, H, nx)
            Y = data[-ny:,:-H].T.reshape(-1, H, ny)
            Y_val = data[-ny:,-H:].T.reshape(-1, H, ny)
            x0 = X[:,[0],:]
            x0_val = X_val[:,[0],:]
        elif 'Daisy' in load_directory:
            U = data[:,1:6]
            Y = data[:,6:9]

            nu = U.shape[1]
            ny = Y.shape[1]
            H = Y.shape[0]

            U = U.reshape(-1, H, nu)
            Y = Y.reshape(-1, H, ny)

            um = np.mean(U, axis=1, keepdims=True)
            us = np.std(U, axis=1, keepdims=True)
            U = (U - um) / us

            ym = np.mean(Y, axis=1, keepdims=True)
            ys = np.std(Y, axis=1, keepdims=True)
            Y = (Y - ym) / ys

            x0 = x0_val = np.zeros((1,1,ny))
            X = X_val = Y
            U_val = U
            Y_val = Y

    scipy.io.savemat(os.path.join(save_directory, f'{save_name}.mat'), 
                     {'U': _squeeze(U), 'U_val': _squeeze(U_val), 'X': _squeeze(X), 'X_val': _squeeze(X_val),
                      'Y': _squeeze(Y), 'Y_val': _squeeze(Y_val), 'x0': _squeeze(x0), 'x0_val': _squeeze(x0_val)})

def save_run(directory, save_name, simba, names, times, train_ids, validation_ids, data, sim_params=None, data_params=None):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    savename = os.path.join(directory, save_name)
    torch.save(
        {
            'model_state_dict': simba.state_dict(),
            'optimizer_state_dict': simba.optimizer.state_dict(),
            'params': simba.params,
            'fit_data': simba.data,
            'names': names,
            'times': times,
            'train_ids': train_ids,
            'validation_ids': validation_ids,
            'sim_params': sim_params,
            'data_params': data_params, 
            'data': data,
        },
        savename,
    )

def load_run(directory, save_name, simba=None):
    # Load the checkpoint
    checkpoint = torch.load(os.path.join(directory, save_name), map_location=lambda storage, loc: storage)

    # Put it into the model
    if simba is not None:
        simba.load_state_dict(checkpoint['model_state_dict'])
        simba.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if torch.cuda.is_available():
            for state in simba.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.tensor):
                        state[k] = v.cuda()
            simba.to(simba.device)
        simba.loaded_params = checkpoint['params']
        simba.loaded_data = checkpoint['fit_data']
        simba.check_loaded_run() 
    names = checkpoint['names']
    times = checkpoint['times']
    train_ids = checkpoint['train_ids']
    validation_ids = checkpoint['validation_ids']
    data = checkpoint['data']
    sim_params = checkpoint['sim_params']
    data_params = checkpoint['data_params']

    return simba, names, times, train_ids, validation_ids, data, sim_params, data_params

def print_all_perf(names, times, train_ids, validation_ids, test_ids, Y_, Y_val_, Y_test_=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if isinstance(Y_, torch.DoubleTensor):
            Y = Y_.cpu().detach().numpy()
            Y_val = Y_val_.cpu().detach().numpy()
            Y_test = Y_test_.cpu().detach().numpy() if Y_test_ is not None else None
        else:
            Y = Y_
            Y_val = Y_val_
            Y_test = Y_test_
        mat = False
        simba = False
        if Y_test is None:
            print('\nMethod\t\tTime\t\tTrain Perf\tVal Perf')
            print('---------------------------------------------------------------')
            for name, time, train, val in zip(names, times, train_ids, validation_ids):
                if (name[:4] == 'mat-') and not mat:
                    print('--------------------------------------------------------')
                    mat = True
                if (name[:4] == 'SIMB') and not simba:
                    print('--------------------------------------------------------')
                    simba = True
                if len(name) < 8:
                    print(f'{name}\t\t{time:.2E}s\t{np.mean((train - Y[0,:,:])**2):.2E}\t{np.mean((val - Y_val[0,:,:])**2):.2E}')
                else:
                    print(f'{name}\t{time:.2E}s\t{np.mean((train - Y[0,:,:])**2):.2E}\t{np.mean((val - Y_val[0,:,:])**2):.2E}')
        else:
            print('\nMethod\t\tTime\t\tTrain Perf\tVal Perf\tTest perf')
            print('------------------------------------------------------------------------')
            for name, time, train, val, test in zip(names, times, train_ids, validation_ids, test_ids):
                if (name[:4] == 'mat-') and not mat:
                    print('------------------------------------------------------------------------')
                    mat = True
                if (name[:4] == 'SIMB') and not simba:
                    print('------------------------------------------------------------------------')
                    simba = True
                if len(name) < 8:
                    print(f'{name}\t\t{time:.2E}s\t{np.mean((train - Y[0,:,:])**2):.2E}\t{np.mean((val - Y_val[0,:,:])**2):.2E}\t{np.mean((test - Y_test)**2):.2E}')
                else:
                    print(f'{name}\t{time:.2E}s\t{np.mean((train - Y[0,:,:])**2):.2E}\t{np.mean((val - Y_val[0,:,:])**2):.2E}\t{np.mean((test - Y_test)**2):.2E}')

def eval_simba(simba, name, names, times, train_ids, validation_ids, test_ids, U, U_val, U_test, X, X_val, X_test, Y, Y_val, Y_test, x0, x0_val, x0_test):

    batch_u, _, batch_x0, batch_y0 = simba.build_data(U, X, Y, x0, [0])
    train_y, _ = simba(batch_u, batch_x0, batch_y0)
    batch_u, _, batch_x0, batch_y0 = simba.build_data(U_val, X_val, Y_val, x0_val, [0])
    val_y, _ = simba(batch_u, batch_x0, batch_y0)
    batch_u, _, batch_x0, batch_y0 = simba.build_data(U_test, X_test, Y_test, x0_test)
    test_y, _ = simba(batch_u, batch_x0, batch_y0)

    names.append(name)
    times.append(simba.times[-1])
    train_ids.append(train_y[0,:,:].cpu().detach().numpy())
    validation_ids.append(val_y[0,:,:].cpu().detach().numpy())
    test_ids.append(test_y[0,:,:].cpu().detach().numpy())

    return names, times, train_ids, validation_ids, test_ids
 
def load_mat(filename):
    """
    Adapted from https://blog.nephics.se/2019/08/28/better-loadmat-for-scipy/
    
    Improved loadmat (replacement for scipy.io.loadmat)
    Ensures correct loading of python dictionaries from mat files.

    Inspired by: https://stackoverflow.com/a/29126361/572908
    """

    def _has_struct(elem):
        """Determine if elem is an array
        and if first array item is a struct
        """
        return isinstance(elem, np.ndarray) and (
            elem.size > 0) and isinstance(
            elem[0], scipy.io.matlab.mat_struct)

    def _check_keys(d):
        """checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            elem = d[key]
            if isinstance(elem,
                          scipy.io.matlab.mat_struct):
                d[key] = _todict(elem)
            elif _has_struct(elem):
                d[key] = _tolist(elem)
        return d

    def _todict(matobj):
        """A recursive function which constructs from
        matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem,
                          scipy.io.matlab.mat_struct):
                d[strg] = _todict(elem)
            elif _has_struct(elem):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        """A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the
        elements if they contain matobjects.
        """
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem,
                          scipy.io.matlab.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif _has_struct(sub_elem):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    data = scipy.io.loadmat(
        filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)
