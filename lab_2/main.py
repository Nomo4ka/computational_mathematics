#Вариант 7

import numpy as np
import matplotlib.pyplot as plt
import math

EPS = 1e-5

def f(x):
    return 1-0.5*np.sin(x)

def g(x):
    return np.acos(0.7-2*x)+1


x_sym , y_sym = sp.symbols('x y')
x = np.linspace(-7,7,100)
plt.figure()
plt.plot(x,f(x),label='$f(x) = \sin(x) + 2y = 2',color='black') 
plot_implicit(Eq(2*x_sym + cos(y_sym-1 ),0.7))
plt.grid()
plt.show()

