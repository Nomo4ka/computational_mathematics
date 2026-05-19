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
    n = 3
    coeffs = []
    C = 0 
    B = (f(xarr[1]) - f(xarr[0]))/h
    for i in range(n):
        xi = xarr[i]
        xi1 = xarr[i+1]
        A = f(xi)
        if i != 0:
            C = (f(xi1) - A - B*h) / h**2
        coeffs.append(ParabCoefficient(A, B, C))
        B = B + 2*C*h
    return coeffs

def Paraspline(x,xarr,coeffs):
    i = np.searchsorted(xarr, x) - 1
    if i < 0:
        i = 0
    if i >= len(coeffs):
        i = len(coeffs) - 1
    c = coeffs[i]
    dx = x - xarr[i]
    return c.A + c.B*dx + c.C*dx**2

def cubicCoeffs(xarr, h):
    n = 3
    coeffs = []
    arr = np.zeros((4, 4))
    arrR = np.zeros(4)
    arr[0][0] = 1 
    arr[3][3] = 1
    for i in range(1,3):
        arr[i][i - 1] = h
        arr[i][i] = 4*h
        arr[i][i+1] = h
        arrR = 3 * (
            (f(xarr[i+1]-f(xarr[i])))/h - (f(xarr[i]) - f(xarr[i-1]))/h
        )
    C = linal
    
def cubicSpline(x,xarr,coeffs):
    

def graph(xarr, h):
    Xdense = np.linspace(xarr[0], xarr[3], 500)
    Y = [f(x) for x in Xdense]
    Ylag = [Lagrange(x, xarr) for x in Xdense]
    Ynew = [Newton(x, xarr, h) for x in Xdense]
    para = coeffscalc(xarr, h)
    Ypara = [Paraspline(x, xarr, para) for x in Xdense]
    plt.plot(Xdense, Y, label='f(x)')
    plt.plot(Xdense, Ylag, label='Lagrange')
    plt.plot(Xdense, Ynew, label='Newton')
    plt.plot(Xdense, Ypara, label='Parabolic Spline')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    h = 0.073
    vecXstr = np.arange(1, 1.219 , h)
    graph(vecXstr, h)
    
if __name__ == "__main__":
    main()
