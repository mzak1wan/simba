
# Parameters to save the model
save_name = 'SIMBa'
directory = 'saves'
train_data = 3
validation_data = 9
test_data = 4

# Define the size of the system (m,n,p)
nu = 5 # m
nx = 4 # n
ny = 3 # p

# Modify to learn discretized continuous systems (Proposition 2)
delta = None


## Import SIMBa and the default parameters
from simba.parameters import base_parameters as parameters
from simba.model import Simba

## Prepare the data (unused data can be set to None,
#  e.g., X for input-output systems)
U, X, Y, x0 = train_data
U_val, X_val, Y_val, x0_val = validation_data
U_test, X_test, Y_test, x0_test = test_data

## Customize the parameters to the problem
parameters['input_output'] = True
# ...
    
## Declare Simba, train it, and save the results
simba = Simba(nx=nx, nu=nu, ny=ny, parameters=parameters)
simba.fit(U, U_val=U_val, U_test=U_test, X=X, X_val=X_val, X_test=X_test, 
          Y=Y, Y_val=Y_val, Y_test=Y_test, x0=x0, x0_val=x0_val, x0_test=x0_test)
simba.save(directory=directory, save_name=save_name)