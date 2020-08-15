README for Fokker Planck Code:
############################################################################################################################################################
Summary:

This code was a final project for a class on Numerical Methods for solving PDEs. The objective of the code was to implement a variety
of different methods in order to solve the Fokker-Planck Equations on a simple test problem and compare the convergence characteristics
and accuracies of a bunch of different methods.

############################################################################################################################################################
File Structure:

This zip file consists of the following files (followed by a short description):
Numerical Solution for Fokker Planck Equations in Accelerators.ipynb (Description and writing the code for the 2OI method)
Fokker Planck Implementation.ipynb (Implementation of the 2OI and Analysis)
Fokker Planck Three Operators.ipynb (Code and testing for the IICN, IIBE, EIBE, EICN methods)
Testing Convergence Three Operators.ipynb (Analysis for IICN, IIBE, EIBE, and EICN methods)
FokkerPlanck.py (Module with methods and comments for implementing a solver)
Fokker_Planck.pdf (Paper that explains the purpose of the project and how each method works and how they performed)
Object Oriented Approach to Simulation.ipynb (Implementation of an Object Oriented Approach to the simulation)


Most of the terminology/purpose is explained in Fokker_Planck.pdf as well as the abbreviations (in section 5,6), the notebooks (section 10). The Object 
Oriented Approach to Simulation illustrates how this code could be implemented to build simulation objects that use object composition. Benefits of this
method are mostly easier use and understanding.
############################################################################################################################################################
Example:

The general structure of the solvers are the following:

"""
for t in time_steps:
    sol_IICN = run_test(0.1,D/t,t,rho(X,Y,0.95),L1 =implicit_L1,L2=implicit_L2,L3=solve_CN,X=X,Y=Y,g=g,rho=rho,t_0=0.95)
    time_error.append(sol_IICN[1][-1])
    time_delta.append(D/t)
"""

run_test will take in initial conditions as well as parameters to solve the problem and then compute the solution of the problem
exactly then computes the error of a method that uses L1, L2, and L3 operators that can be set as kwargs. These "operators" are
functions that are passed as arguments which makes it very easy to create a solve by using different functions that can be mixed
and matched as we aim for more modular framework that can be used to test all combinations of these operators in the accompanying
paper analysis.
############################################################################################################################################################
