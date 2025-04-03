# Prelim
import numpy as np
import itertools

#---------------------------------------------------------------------------------------------------

# ODEs

def Equ1(x0, e1):
    V_in = 1
    e0 = 0.0467
    k_cat = 12
    k_m = 10
    lam = 1.93E-4
    term1 = V_in
    term2 = e0 * (k_cat * x0) / (k_m + x0)
    term3 = e1 * (k_cat * x0) / (k_m + x0)
    term4 = lam * x0
    return term1 - term2 - term3 - term4
    
def Equ2(x0, x1, e1, e2):
    k_cat = 12
    k_m = 10
    lam = 1.93E-4
    term1 = e1 * (k_cat * x0) / (k_m + x0)
    term2 = e2 * (k_cat * x1) / (k_m + x1)
    term3 = lam * x1
    return term1 - term2 - term3
    
def Equ3(x1, e1, k1, theta1):
    lam = 1.93E-4
    u = k1 * theta1**2 / (theta1**2 + x1**2)
    return u - lam * e1
    
def Equ4(x1, e2, k2, theta2):
    lam = 1.93E-4
    u = k2 * theta2**2 / (theta2**2 + x1**2)
    return u - lam * e2

#---------------------------------------------------------------------------------------------------

# Sensitivity functions
def S_k1_x1_analytic(x0, x1, e1, e2, k1, k2, theta1, theta2):

    # Constants:
    V_in = 1
    e0 = 0.0467
    k_cat = 12
    k_m = 10
    lam = 1.93E-4
    x0prime = Equ1(x0, e1)
    
    # Numerator:
    num = k1 * lam * theta1**2 * (
        theta1**2 * (k1 * x0prime * (e0 * lam + k1) - e0 * V_in)
        + x1**2 * (e0 * k1 * lam * x0prime - e0 * V_in)
        + e0 * lam * x0 * (theta1**2 + x1**2))
    
    # Denominator:
    factor1 = x1 * (theta1**2 * (e0 * lam + k1) + e0 * lam * x1**2)**2
    term1 = (2 * k1**2 * theta1**4 * x1 * (V_in - lam * x0)) / (
                (theta1**2 + x1**2) * (theta1**2 * (e0 * lam + k1) + e0 * lam * x1**2)**2)
    term2 = -(2 * k1 * theta1**2 * x1 * (V_in - lam * x0)) / (
                (theta1**2 + x1**2) * (theta1**2 * (e0 * lam + k1) + e0 * lam * x1**2))
    term3 = -(k2 * k_cat * theta2**2) / (lam * (k_m + x1) * (theta2**2 + x1**2))
    term4 = (k2 * k_cat * theta2**2 * x1) / (lam * (k_m + x1)**2 * (theta2**2 + x1**2))
    term5 = (2 * k2 * k_cat * theta2**2 * x1**2) / (lam * (k_m + x1) * (theta2**2 + x1**2)**2)
    term6 = -lam
    factor2 = term1 + term2 + term3 + term4 + term5 + term6
    denom = factor1 * factor2

    sensitivity = num / denom
    return np.log10(abs(sensitivity))


def S_k2_x1_analytic(x0, x1, e1, e2, k1, k2, theta1, theta2):

    # Constants:
    V_in = 1
    e0 = 0.0467
    k_cat = 12
    k_m = 10
    lam = 1.93E-4
    x0prime = Equ1(x0, e1)
    
    # Numerator: 
    num = k2 * (
        (k1 * lam * theta1**2 * x0prime) / (theta1**2 * (e0 * lam + k1) + e0 * lam * x1**2)
        +
        (k_cat * theta2**2 * x1) / (lam * (k_m + x1) * (theta2**2 + x1**2)))
        
    # Denominator:
    term1 = (2 * k1**2 * theta1**4 * x1 * (V_in - lam * x0)) / (
                (theta1**2 + x1**2) * (theta1**2 * (e0 * lam + k1) + e0 * lam * x1**2)**2)
    term2 = -(2 * k1 * theta1**2 * x1 * (V_in - lam * x0)) / (
                (theta1**2 + x1**2) * (theta1**2 * (e0 * lam + k1) + e0 * lam * x1**2))
    term3 = -(k2 * k_cat * theta2**2) / (lam * (k_m + x1) * (theta2**2 + x1**2))
    term4 = (k2 * k_cat * theta2**2 * x1) / (lam * (k_m + x1)**2 * (theta2**2 + x1**2))
    term5 = (2 * k2 * k_cat * theta2**2 * x1**2) / (lam * (k_m + x1) * (theta2**2 + x1**2)**2)
    term6 = -lam
    denom = x1 * (term1 + term2 + term3 + term4 + term5 + term6)
    
    sensitivity = num / denom
    return np.log10(abs(sensitivity))


#---------------------------------------------------------------------------------------------------

# Initial guesses

def generate_initial_guesses():
    pos1 = [0, 1000, 2000]
    pos2 = [0, 50, 100, 150, 200, 250, 300]
    pos3 = [0, 2.5, 5]
    pos4 = [0, 2.5, 5]

    combinations = itertools.product(pos1, pos2, pos3, pos4)
    return [np.array(comb) for comb in combinations]
