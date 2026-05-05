import numpy as np 
import matplotlib.pyplot as plt

def f(x):
    return x/(10*np.pi * np.sin(x))

#max(abs(f(x) - P_{n}(x)) le dfrac{M_{n+1}}{(n+1)!} * h
