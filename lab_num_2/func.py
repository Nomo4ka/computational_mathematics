import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

x = sp.symbols('x')

def quad(x):
    return sp.diff(f(x), x, 4)

def f(x):
    return x/(10*np.pi * sp.sin(x))

quad = quad(x)



print(quad)