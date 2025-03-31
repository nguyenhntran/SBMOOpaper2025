# Prelim

import numpy as np

# ODEs

#def Equ1(x, alpha, n):
#    return (alpha / (1 + x**n)) - x

# Sensitivity functions

def S_kE1_P_analytic():
    return float(1)
    
def S_kE2_P_analytic():
    return float(1)

def S_kE1_S2_analytic():
    return float(1)

def S_kE2_S2_analytic():
    return float(0)
    
def S_kE1_ATP_analytic():
    return float(1)

def S_kE2_ATP_analytic(kS2, kE2, alphaE, gammaE):
    frac = (kE2/kS2) * (alphaE/gammaE)
    return frac / (1 + frac)

# Initial guesses

#def generate_initial_guesses(alpha_val, n_val):
#    return [
#        np.array([2]),
#        np.array([0.5]),
#        np.array([4.627])
#    ]