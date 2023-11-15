import time
import numpy as np

from torch import nn
import torch
from torch import optim
import torch.nn.functional as F

from slode.util import check_and_initialize_data, generate_A_Hurwitz

class SLODE(nn.Module):
    
    def __init__(self, nx, nu, ny, A_init=None, B_init=None, C_init=None, D_init=None, id_D=False, stable_A=True, LMI_A=True, tol_A=1e-9, input_output=False, learn_x0=True):
        super().__init__()
        
        if id_D:
            assert input_output, 'Cannot identify "D" on input state models'
        
        self.nx = nx
        self.nu = nu
        self.ny = ny

        self.id_D = id_D
        
        self.stable_A = stable_A
        self.LMI_A = LMI_A
        self.tol_A = tol_A
        
        self.input_output = input_output
        self.learn_x0 = learn_x0
        
        if self.stable_A:
            if self.LMI_A:
                self.M = nn.Parameter(torch.randn(2*nx, 2*nx), True)
            else:
                self.M = nn.Parameter(torch.randn(nx, nx), True)
                self.A_ = nn.Parameter(torch.randn(nx, nx), True)
        else:
            if A_init is None:
                self.A_ = nn.Parameter(torch.FloatTensor(generate_A_Hurwitz(nx)), True)
            else:
                self.A_ = nn.Parameter(torch.FloatTensor(A_init), True)
        
        if B_init is None:
            self.B = nn.Parameter(torch.randn(nx,nu), True)
        else:
            self.B = nn.Parameter(torch.FloatTensor(B_init), True)
        
        if self.input_output:
            if C_init is None:
                self.C = nn.Parameter(torch.randn(ny,nx), True)
            else:
                self.C = nn.Parameter(torch.FloatTensor(C_init), True)
                
        if self.id_D:
            if self.input_output:
                if D_init is None:
                    self.D = nn.Parameter(torch.randn(ny,nu), True)
                else:
                    self.D = nn.Parameter(torch.FloatTensor(C_init), True)
            
        if self.learn_x0:
            self.x0 =  nn.Parameter(torch.randn(nx,1), True)
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("\nGPU acceleration on!")
        else:
            self.device = "cpu"
        self.to(self.device)
        
    @property
    def A(self):
        if self.stable_A:
            if self.LMI_A:
                S = self.M.clone().matmul(self.M.clone().T) + self.tol_A * torch.eye(2*self.nx, requires_grad=True)
                F = S[self.nx:,:self.nx]
                P = S[self.nx:,self.nx:]
                E = 0.5*(P + S[:self.nx,:self.nx])
                A_ = torch.inverse(E).matmul(F)
            else:
                M_ = 1 - 0.1*torch.sigmoid(self.M.clone())
                A_ = self.A_.clone()
                for row in range(A_.shape[0]):
                    sum_ = torch.sum(torch.exp(A_[row, :]))
                    A_[row, :] = torch.exp(A_[row, :]) / sum_ * M_[row, :]
            return A_
        else:
            return self.A_
                
    def forward(self, u_, x0=None):
                
        if self.learn_x0:
            x = torch.stack([self.x0.clone() for _ in range(u_.shape[0])], dim=0)
        else:
            assert x0 is not None, 'If x0 is not learned, an itial point must be given to the forward pass'
            x = x0.clone().permute(0,2,1)
                        
        u = u_.clone()
        
        if len(u.shape) == 1:
            u = u.unsqueeze(0)
            u = u.unsqueeze(-1)
        elif len(u.shape) == 2:
            u = u.unsqueeze(0)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        if self.input_output:
            predictions = torch.empty((u.shape[0], u.shape[1], self.ny))
            C = torch.stack([self.C.clone() for _ in range(u.shape[0])], dim=0)
            if self.id_D:
                D = torch.stack([self.D.clone() for _ in range(u.shape[0])], dim=0)
                predictions[:,0,:] = (torch.bmm(C, x.clone()) + torch.bmm(D, u[:,0,:].unsqueeze(-1))).squeeze(-1).clone()
            else:
                predictions[:,0,:] = torch.bmm(C, x.clone()).squeeze(-1).clone()
        else:
            predictions = torch.empty((u.shape[0], u.shape[1], self.nx))
            predictions[:,0,:] = x.squeeze(-1).clone()
                    
        for t in range(u.shape[1]-1):
            
            A = torch.stack([self.A.clone() for _ in range(u.shape[0])], dim=0)
            B = torch.stack([self.B.clone() for _ in range(u.shape[0])], dim=0)
            x = torch.bmm(A, x) + torch.bmm(B, u[:,t,:].unsqueeze(-1))
            
            if self.input_output:
                C = torch.stack([self.C.clone() for _ in range(u.shape[0])], dim=0)
                if self.id_D:
                    D = torch.stack([self.D.clone() for _ in range(u.shape[0])], dim=0)
                    predictions[:,t+1,:] = (torch.bmm(C, x.clone()) + torch.bmm(D, u[:,t,:].unsqueeze(-1))).squeeze(-1).clone()
                else:
                    predictions[:,t+1,:] = torch.bmm(C, x.clone()).squeeze(-1).clone()
            else:
                predictions[:,t+1,:] = x.squeeze(-1).clone()
            
        return predictions
        
        
class SLODEWrapper(SLODE):

    def __init__(self, nx, nu, ny, A_init=None, B_init=None, C_init=None, D_init=None, id_D=False, stable_A=False, LMI_A=True, tol_A=1e-9, input_output=False, learn_x0=True, lr=0.1, batch_size=4, verbose=1):
        
        super().__init__(nx, nu, ny, A_init, B_init, C_init, D_init, id_D, stable_A, LMI_A, tol_A, input_output, learn_x0)
        
        self.lr = lr
        self.batch_size = batch_size
        self.verbose = verbose
        
        self.times = []
        self.train_losses = []
        self.val_losses = []
        self.auto_fit = False
        
        self.loss = F.mse_loss
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
                                
    def batch_iterator(self, U, batch_size: int = None, shuffle: bool = True) -> None:

        if batch_size is None:
            batch_size = self.batch_size
            
        indices = np.arange(U.shape[0])
            
        # Shuffle them if wanted
        if shuffle:
            np.random.shuffle(indices)

        # Define the right number of batches according to the wanted batch_size - taking care of the
        # special case where the indicies ae exactly divisible by the batch size, which can induce
        # an additional empty batch breaking the simulation down the line
        n_batches = int(np.ceil(len(indices) / batch_size))

        # Iterate to yield the right batches with the wanted size
        for batch in range(n_batches):
            yield indices[batch * batch_size: (batch + 1) * batch_size]
            
    def build_data(self, U, X=None, Y=None, x0=None, indices=None):
    
        if indices is None:
            indices = [0]
                    
        u = torch.FloatTensor(U[indices, :, :]).to(self.device)
            
        if self.input_output:
            assert Y is not None, 'Y must be provided for input output identification'
            y = torch.FloatTensor(Y[indices, :, :]).to(self.device)
            if not self.learn_x0:
                assert x0 is not None, 'x0 must be provided if not learned'
                x0 = torch.FloatTensor(x0[indices, :, :]).to(self.device)
            else:
                x0 = None
            return u, y, x0
                
        else:
            assert X is not None, 'X must be provided for input state identification'
            x = torch.FloatTensor(X[indices, :, :]).to(self.device)
            if not self.learn_x0:
                x0 = torch.FloatTensor(X[indices, [0], :]).to(self.device)
            else:
                x0 = None
            return u, x, x0
            
    def fit(self, U, U_val=None, X=None, X_val=None, Y=None, Y_val=None, x0=None, x0_val=None, max_epochs=150, return_best=True, tol=1e-12, print_each=25):
        
        U, U_val, X, X_val, Y, Y_val, x0, x0_val = check_and_initialize_data(U, U_val, X, X_val, Y, Y_val, x0, x0_val, verbose=self.verbose, input_output=self.input_output)

        if (self.verbose > 0) and not self.auto_fit:
            print("\nTraining starts!\n")
        
        if len(self.val_losses) > 0:
            self.best_loss = np.min(self.val_losses)
        else:
            self.best_loss = np.inf
        
        self.times.append(time.time())
            
        # Assess the number of epochs the model was already trained on to get nice prints
        trained_epochs = len(self.train_losses)

        for epoch in range(trained_epochs, trained_epochs + max_epochs):
        
            if epoch == print_each-1:
                print('Epoch\tTraining loss\tValidation loss')
            
            self.train()
            train_losses = []
            for indices in self.batch_iterator(U=U):

                self.optimizer.zero_grad()
                
                # Compute the loss of the batch and store it
                batch_u, batch_data, batch_x0 = self.build_data(U, X, Y, x0, indices)
                predictions = self.forward(x0=batch_x0, u_=batch_u)
                loss = self.loss(predictions, batch_data)
                    
                # Compute the gradients and take one step using the optimizer
                loss.backward()
                self.optimizer.step()
                
                train_losses.append(float(loss))
                
            self.train_losses.append(np.mean(train_losses))
            
            self.eval()
            val_losses = []
            for indices in self.batch_iterator(U=U_val):
            
                batch_u, batch_data, batch_x0 = self.build_data(U_val, X_val, Y_val, x0_val, indices)
                predictions = self.forward(x0=batch_x0, u_=batch_u)
                loss = self.loss(predictions, batch_data)
                    
                val_losses.append(float(loss))
                
            self.val_losses.append(np.mean(val_losses))
                
            # Compute the average loss of the training epoch and print it
            if (self.verbose > 0) and (epoch % print_each == print_each-1):
                print(f"{epoch + 1}\t{self.train_losses[-1]:.2E}\t{self.val_losses[-1]:.2E}")
                
            if float(loss) < self.best_loss:
                self.best_loss = float(loss)
                self.save_state()
                
            elif float(loss) > np.array(self.val_losses[-20:]).mean() - tol:
                if self.auto_fit and (epoch <= 10):
                    raise ValueError('A very specific bad thing happened.')
                elif self.verbose > 0:
                    print(f'\nStopping training, no improvement according to the tolerance set at {tol:.0E}')
                break
            
            elif np.isnan(float(loss)):
                print('\nEplosion! Restarting training\n')
                self.__init__(nx, nu, ny, id_D=self.id_D, stable_A=self.stable_A, LMI_A=self.LMI_A, tol_A=self.tol_A, input_output=self.input_output, learn_x0=self.learn_x0, lr=self.lr, batch_size=self.batch_size)
                self.fit(U, U_val, X, X_val, Y, Y_val, x0, x0_val, max_epochs, return_best, tol, print_each)
                if self.auto_fit:
                    raise ValueError('A very specific bad thing happened.')
                break
                
            # Timing information
            self.times.append(time.time())

        if return_best:
            self.overwrite_best_performance()
            
        if self.verbose > 0:
            print(f"\nAverage time per epoch:\t{np.diff(self.times).mean():.2f}s")
            print(f"Total training time:\t{int(self.times[-1] - self.times[trained_epochs])}s")
            print(f"Best loss at epoch {np.argmin(self.val_losses)}:\t{np.min(self.val_losses):.2E}")
            
        if self.verbose > 1:
            print('\nFirst elements of each matrices')
            print(f'A:\n{self.A[:5,:5].detach().numpy()}')
            print(f'B:\n{self.B[:5,:5].detach().numpy()}')
            if self.input_output:
                print(f'C:\n{self.C[:5,:5].detach().numpy()}')
            if self.id_D:
                print(f'D:\n{self.D[:5,:5].detach().numpy()}')

    def save_state(self):
        self.best_state = self.state_dict()
        self.best_optimizer = self.optimizer.state_dict()
            
    def overwrite_best_performance(self):
        self.load_state_dict(self.best_state)
        self.optimizer.load_state_dict(self.best_optimizer)

def Slode(nx, nu, ny, A_init=None, B_init=None, C_init=None, D_init=None, id_D=False, stable_A=False, LMI_A=True, tol_A=1e-9, input_output=False, learn_x0=True, lr=0.1, batch_size=4, verbose=4):
    return SLODEWrapper(nx=nx, nu=nu, ny=ny, A_init=A_init, B_init=B_init, C_init=C_init, D_init=D_init, id_D=id_D, stable_A=stable_A, LMI_A=LMI_A, tol_A=tol_A, input_output=input_output, learn_x0=learn_x0, lr=lr, batch_size=batch_size, verbose=verbose)

def Slode_auto_fit(nx, nu, ny, A_init=None, B_init=None, C_init=None, D_init=None, id_D=False, stable_A=False, LMI_A=True, tol_A=1e-9, input_output=False, learn_x0=True, lr=0.1, batch_size=4, verbose=4, U=None, U_val=None, X=None, X_val=None, Y=None, Y_val=None, x0=None, x0_val=None, max_epochs=150, return_best=True, tol=1e-12, print_each=25):

    assert U is not None, 'U has to be provided'
    
    U, U_val, X, X_val, Y, Y_val, x0, x0_val = check_and_initialize_data(U, U_val, X, X_val, Y, Y_val, x0, x0_val, verbose=verbose, input_output=input_output)

    print('')

    for lr in [2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
        try:
            print(f'Trying with learning rate {lr:.0E}\n')
            slode = Slode(nx, nu, ny, id_D=id_D, stable_A=stable_A, LMI_A=LMI_A, tol_A=tol_A, input_output=input_output, learn_x0=learn_x0, lr=lr, batch_size=batch_size)
            slode.auto_fit = True
            slode.fit(U=U, U_val=U_val, X=X, X_val=X_val, Y=Y, Y_val=Y_val, x0=x0, x0_val=x0_val, max_epochs=max_epochs, return_best=return_best, tol=tol, print_each=print_each)
            break
        except ValueError:
            continue
            
    return slode
