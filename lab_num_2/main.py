import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from math import factorial as fact

#[1,3]
EPS = 1e-5

def f(x):
    return x/(10*np.pi * np.sin(x))

@dataclass
class Coefficients:
    A: float
    B: float
    C: float

@dataclass
class ParabCoefficient(Coefficients):
    pass

@dataclass
class CubicCoefficient(Coefficients):
    D: float
    
def Lagrange(x, xarr):
    res = 0
    for i in range(4):
        li = 1
        for j in range(4):
            if i != j:
                li *= (x-xarr[j])/(xarr[i]-xarr[j])
        res += f(xarr[i]) * li
    return res

def NewtonTable(xarr):
    table = np.zeros((4, 4))
    for i in range(4):
        table[i][0] = f(xarr[i])
        for j in range(1, 4):
            for k in range(4-j):
                table[k][j] = (table[k+1][j-1] - table[k][j-1]) 
    return table

def Newton(x,xarr,h):
    t = (x - xarr[0]) / h
    table = NewtonTable(xarr)
    res = table[0][0]
    for j in range(1, 4):
        term = table[0][j]
        for k in range(j):
            term *= (t - k)/fact(k+1)
        res += term
    return res

def coeffscalc(xarr, h):
    n = 4
    coeffs = []
    C = 0 
    B = (f(xarr[1] - f(xarr[0])))/h
    for i in range(n):
        xi = xarr[i]
        xi1 = xarr[i+1]
        A = f(xi)
        if i != 0:
            C = (f(xi1) - 2*f(xi) + f(xi-1)) / h**2
        coeffs.append(ParabCoefficient(A, B, C))
        B = B + 2*C*h
    return coeffs

def Paraspline(x,xarr,paracoeffs):
    return paracoeffs.A + paracoeffs.B*(x-xarr[0]) + paracoeffs.C*(x-xarr[0])**2

def graph(xarr, h):
    Y = [f(x) for x in xarr]
    Ylag = [Lagrange(x, xarr) for x in xarr]
    Ynew = [Newton(x, xarr, h) for x in xarr]
    para = coeffscalc(xarr, h)
    Ypara = [Paraspline(x, xarr, para[0]) for x in xarr]
    
    plt.plot(xarr, Y, label='f(x)')
    plt.plot(xarr, Ylag, label='Lagrange')
    plt.plot(xarr, Ynew, label='Newton')
    plt.plot(xarr, Ypara, label='Parabolic Spline')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    h = 0.073
    vecXstr = np.arange(1, 1.219 , h)
    graph(vecXstr, h)
    
if __name__ == "__main__":
    main()
