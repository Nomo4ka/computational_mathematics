import sympy as sp
from scipy.optimize import minimize_scalar
from math import cos, sin, pi
import numpy as np
from math import factorial

x = sp.Symbol('x')
def g(x):
    return x/(10*sp.pi * sp.sin(x))

def Ng(x):
    return x/(10*np.pi * np.sin(x))

def f(x):
    return (x*(5 + 28*cos(x)**2/sin(x)**2 + 24*cos(x)**4/sin(x)**4) - 4*(5 + 6*cos(x)**2/sin(x)**2)*cos(x)/sin(x))/(10*pi*sin(x))

res = minimize_scalar(lambda x: -f(x), bounds=(1, 3), method="bounded")

candidates = [(1, f(1)), (3, f(3)), (res.x, f(res.x))]
x_max, y_max = max(candidates, key=lambda p: p[1])

print(x_max, y_max)

print(f"в начале 1 = {Ng(1)} и 3 = {Ng(3)}")

n = 3
eps = 5e-6
M = y_max

h = pow(factorial(n + 1) * eps / M,1/4)
#Предполгаем , что h = 0.01
print("h =", h)
