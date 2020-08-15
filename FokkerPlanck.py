import matplotlib.pyplot as plt
import numpy as np
import numpy
from scipy import signal

def run_test(delta_x, delta_t, time_steps, initial_condition, L1=None, L2 = None, L3=None, X=None, Y=None, g=None,rho=None, kappa=.8,t_0=0):
    """
    Code meant to combine L1, L2, and L3 steps in order to simulate the Fokker Planck Equations
    
    :Input:
     - *delta_x* (float) Distance between points in discretization
     - *delta_t* (float) Length of a Time Step
     - *time_steps* (int) Number of time steps
     - *initial_condition* (numpy.ndarray) initial condition
     - *L1* (function) function that represents the L1 operator
     - *L2* (function) function that represents the L2 operator
     - *L2* (function) function that represents the L3 operator
     - *X* (numpy.ndarray) Represents the underlying x1 variable which is involved in the process
     - *Y* (numpy.ndarray) Represents the underlying x2 variable which is involved in the process
     - *g* (function) Represents a function on x1 and x2 that is involved with the L2 term
     - *kappa* (float) Term that controls the diffusion in the L3 operator
     - *t_0* starting time
     
    :Output:
     - *U* (numpy.ndarray) Solution
    """
    
    print("Beginning Test")
    U=initial_condition.copy()
    errors=[]
    
    dim1, dim2 = X.shape
    t=t_0
    
    for i in range(time_steps):
        t+=delta_t
        U = L3(U,delta_x,delta_t,kappa=kappa)
        U = L1(U,delta_x,delta_t,Y=Y)
        U = L2(U,delta_x,delta_t,X=X,Y=Y,g=g)
       
        if i%100==0:
            print("@ Timestep {}".format(i))
            
        error=np.sqrt(np.sum((U-rho(X,Y,t))*(U-rho(X,Y,t)))/(dim1*dim2))
        errors.append(error)
    
    return U,errors

#################################################################################################################

def explicit_L3(U,delta_x, delta_t,kappa=.8):
    """
    Will represent an second derivative operator (L3) that will take the solution forward one time step with
    a diffusion constant D, delta_x, and delta_t implemented with an explicit scheme with boundary conditions
    set to zero.
    
    :Input:
     - *U* (numpy.ndarray) Input array
     - *delta_x* (float) Distance between points in discretization
     - *delta_t* (float) Length of a Time Step
     - *kappa* (float) Diffusion Constant
     
    :Output:
     - (numpy.ndarray) Solution at one time step forward.
    """
    x2_dim= U.shape[0]
    r = (kappa* delta_t / (2*delta_x**2))
    d = numpy.ones(x2_dim)
    
    A = numpy.diag((1-2*r)*d) + numpy.diag(r*d[1:],1)+ numpy.diag(r*d[1:],-1)
    
    for i in range(x2_dim):
        U[:,i] = numpy.dot(A,U[:,i])

    return U

def solve_CN(U,delta_x, delta_t,kappa=.8):
    """Will represent an second derivative operator (L3) that will take the solution forward one time step with
    a diffusion constant D, delta_x, and delta_t implemented with a Crank-Nicholson Scheme with boundary conditions
    set to zero.
    
    :Input:
     - *U* (numpy.ndarray) Input array
     - *delta_x* (float) Distance between points in discretization
     - *delta_t* (float) Length of a Time Step
     - *kappa* (float) Diffusion Constant
     
    :Output:
     - (numpy.ndarray) Solution at one time step forward.
    """
    x2_dim= U.shape[0]
    r = (kappa* delta_t / (2*(delta_x**2)))
    d = numpy.ones(x2_dim)
    
    A = numpy.diag((1+2*r)*d) + numpy.diag(-r*d[1:],1)+ numpy.diag(-r*d[1:],-1)
    B = numpy.diag((1-2*r)*d) + numpy.diag(r*d[1:],1)+ numpy.diag(r*d[1:],-1)
    
    for i in range(x2_dim):
        f = numpy.dot(B,U[:,i])
        U[:,i] = numpy.linalg.solve(A,f)

    return U

def solve_BE(U,delta_x, delta_t,kappa=.8):
    """Will represent an second derivative operator (L3) that will take the solution forward one time step with
    a diffusion constant D, delta_x, and delta_t implemented with a Backwards-Euler Scheme with boundary conditions
    set to zero.
    
    :Input:
     - *U* (numpy.ndarray) Input array
     - *delta_x* (float) Distance between points in discretization
     - *delta_t* (float) Length of a Time Step
     - *kappa* (float) Diffusion Constant
     
    :Output:
     - (numpy.ndarray) Solution at one time step forward.
    
    """
    
    x2_dim= U.shape[0] 
    
    r = (kappa* delta_t / (delta_x**2))
    d = numpy.ones(x2_dim)
    A = numpy.diag((1+2*r)*d) + numpy.diag(-r*d[1:],1)+ numpy.diag(-r*d[1:],-1)
    
    for i in range(x2_dim):
        U[:,i] = numpy.linalg.solve(A,U[:,i])
#         U[i+1,0] = U[i,0] + kappa*delta_t/delta_x**2 * (g_0 - 2.0*U[i,0]+U[i,1])
#         U[i+1,-1] = U[i,-1] + kappa*delta_t/delta_x**2 * (U[i,-2] - 2.0*U[i,-1]+g_1)
        
    return U

import scipy.sparse as sparse
import scipy.sparse.linalg as linalg

def solve_BE_mod(U,delta_x,delta_t,kappa=.8):
    """Will represent an second derivative operator (L3) that will take the solution forward one time step with
    a diffusion constant D, delta_x, and delta_t implemented with a Backwards-Euler Scheme with boundary conditions
    set to zero. This one also uses a Scipy based implementation.
    
    :Input:
     - *U* (numpy.ndarray) Input array
     - *delta_x* (float) Distance between points in discretization
     - *delta_t* (float) Length of a Time Step
     - *kappa* (float) Diffusion Constant
     
    :Output:
     - (numpy.ndarray) Solution at one time step forward.
    
    """
    x2_dim= U.shape[0] 
    
    r = (kappa* delta_t / (delta_x**2))
    e = numpy.ones(x2_dim)*r
    D2 = sparse.spdiags([e,-2.0*e,e],[-1,0,1],x2_dim,x2_dim).tocsr()
    I = sparse.eye(x2_dim).tocsr()
    A1 = (I-D2)
    
    for i in range(x2_dim):
        U[:,i] = linalg.spsolve(A1,U[:,i])
#         U[i+1,0] = U[i,0] + kappa*delta_t/delta_x**2 * (g_0 - 2.0*U[i,0]+U[i,1])
#         U[i+1,-1] = U[i,-1] + kappa*delta_t/delta_x**2 * (U[i,-2] - 2.0*U[i,-1]+g_1)
        
    return U

def solve_CN_mod(U,delta_x,delta_t,kappa=.8):
    """Will represent an second derivative operator (L3) that will take the solution forward one time step with
    a diffusion constant D, delta_x, and delta_t implemented with a Crank-Nicholson Scheme with boundary conditions
    set to zero. This one also uses a scipy based implementation.
    
    :Input:
     - *U* (numpy.ndarray) Input array
     - *delta_x* (float) Distance between points in discretization
     - *delta_t* (float) Length of a Time Step
     - *kappa* (float) Diffusion Constant
     
    :Output:
     - (numpy.ndarray) Solution at one time step forward.
    """
    x2_dim= U.shape[0] 
    
    r = (0.5*kappa* delta_t / (delta_x**2))
    e = numpy.ones(x2_dim)*r
    D2 = sparse.spdiags([e,-2.0*e,e],[-1,0,1],x2_dim,x2_dim).tocsr()
    I = sparse.eye(x2_dim).tocsr()
    A1 = (I-D2)
    A2 = (I+D2).tolil()
    
    for i in range(x2_dim):
        f = A2.dot(U[:,i].copy())
        U[:,i] = linalg.spsolve(A1,f)
#         U[i+1,0] = U[i,0] + kappa*delta_t/delta_x**2 * (g_0 - 2.0*U[i,0]+U[i,1])
#         U[i+1,-1] = U[i,-1] + kappa*delta_t/delta_x**2 * (U[i,-2] - 2.0*U[i,-1]+g_1)
        
    return U
################################################################################################################

def implicit_L1(U,delta_x,delta_t,Y=None):
    """Will represent a first derivative operator (L1) that will take the solution forward one time step with
    a delta_x, and delta_t with an implicit implementation.

    :Input:
    - *U* (numpy.ndarray) Input array
    - *delta_x* (float) Distance between points in discretization
    - *delta_t* (float) Length of a Time Step
    - *Y* (numpy.ndarray) Represents the underlying x2 variable which is involved in the process

    :Output:
    - (numpy.ndarray) Solution at one time step forward.
    """
    x2_dim= U.shape[1] 
    for i in range(x2_dim):
        temp=Y[i].copy()
        diag = (delta_t*temp[:-1])/(2*delta_x)
        B = np.eye(x2_dim)+np.diag(diag,1)-np.diag(diag,-1)

        U[i]=np.linalg.solve(B,U[i])

    return U

def explicit_L1(U,delta_x,delta_t,Y=None):
    """Will represent a first derivative operator (L1) that will take the solution forward one time step with
    a delta_x, and delta_t with an explicit implementation.
    
    :Input:
     - *U* (numpy.ndarray) Input array
     - *delta_x* (float) Distance between points in discretization
     - *delta_t* (float) Length of a Time Step
     - *Y* (numpy.ndarray) Represents the underlying x2 variable which is involved in the process
     
    :Output:
     - (numpy.ndarray) Solution at one time step forward.
    """
    
    x2_dim= U.shape[1] 
    for i in range(x2_dim):
        diag = (delta_t*Y[i].copy())/(2*delta_x)
        temp = np.pad(U[i].copy(),(1,1),'constant')
        U[i]-=diag*(temp[2:]-temp[:-2])
        
    return U

def implicit_L2(U,delta_x,delta_t,X=None,Y=None,g=None):
    """Will represent a first derivative operator (L2) that will take the solution forward one time step with
    a delta_x, and delta_t with an implicit implementation.
    
    :Input:
     - *U* (numpy.ndarray) Input array
     - *delta_x* (float) Distance between points in discretization
     - *delta_t* (float) Length of a Time Step
     - *X* (numpy.ndarray) Represents the underlying x1 variable which is involved in the process
     - *Y* (numpy.ndarray) Represents the underlying x2 variable which is involved in the process
     - *g* (function) Represents a function on x1 and x2 that is involved with the L2 term
     
    :Output:
     - (numpy.ndarray) Solution at one time step forward.
    """
    x1_dim= U.shape[0] 
    for i in range(x1_dim):
        top_diag = ((delta_t*g(X[:,i],Y[:,i]))/(2*delta_x))[1:]
        bot_diag = ((delta_t*g(X[:,i],Y[:,i]))/(2*delta_x))[:-1]
        B = np.eye(x1_dim)-np.diag(top_diag,1)+np.diag(bot_diag,-1)
        B[0,1]=0
        B[-1,-2]=0
        U[:,i]=np.linalg.solve(B,U[:,i])
        
    return U


def explicit_L2(U,delta_x,delta_t,X=None,Y=None,g=None):
    """Will represent a first derivative operator (L2) that will take the solution forward one time step with
    a delta_x, and delta_t with an explicit implementation.
    
    :Input:
     - *U* (numpy.ndarray) Input array
     - *delta_x* (float) Distance between points in discretization
     - *delta_t* (float) Length of a Time Step
     - *X* (numpy.ndarray) Represents the underlying x1 variable which is involved in the process
     - *Y* (numpy.ndarray) Represents the underlying x2 variable which is involved in the process
     - *g* (function) Represents a function on x1 and x2 that is involved with the L2 term
     
    :Output:
     - (numpy.ndarray) Solution at one time step forward.
    """
    x1_dim= U.shape[0] 
    for i in range(x1_dim):
        temp = np.pad(U[:,i].copy(),(1,1),'constant')
        diag = ((delta_t*g(X[:,i],Y[:,i]))/(2*delta_x))
        U[:,i]+=diag*(temp[2:]-temp[:-2])
        
    return U