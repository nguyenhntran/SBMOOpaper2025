# Prelim

import numpy as np

# ODEs

def Equ1(A, alpha_A, I_1, K_A, n, \delta_A):
    return \alpha_A / (1+(I_1 / K_A)^n) ​​− \delta_A * ​A #dAdt
    
def Equ2(B, alpha_B, I_2, K_B, n, \delta_B):
    return \alpha_B / (1+(I_2 / K_B)^n) ​​− \delta_B * B #dBdt
    
def Equ3(A, B, alpha_X, K_X, K_Y, m, \delta_X):
    return \alpha_X / (1 + (A / K_X)**m * (B / K_Y)**m) - \delta_X * X #dXdt

# Sensitivity functions

def S_alphaA_A_analytic():
    return 

def S_I1_A_analytic():
    return 

def S_KA_A_analytic():
    return 
    
def S_n_A_analytic():
    return 

def S_deltaA_A_analytic():
    return 

def S_alphaB_B_analytic():
    return 

def S_I2_B_analytic():
    return 

def S_KB_B_analytic():
    return 
    
def S_n_B_analytic():
    return 

def S_deltaB_B_analytic():
    return 

def S_alphaX_X_analytic():
    return 
    
def S_KX_X_analytic():
    return 

def S_KY_X_analytic():
    return 
    
def S_m_X_analytic():
    return 
    
def S_deltaB_X_analytic():
    return 
    
# Initial guesses

def generate_initial_guesses():
    return [
        np.array([... , ... , ...]),
        np.array([... , ... , ...]),
        np.array([... , ... , ...])
    ]