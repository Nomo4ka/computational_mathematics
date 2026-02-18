import math
import numpy as np
import matplotlib.pyplot as plt

#Вариант 7

EPS =1e-3

def f(x):
    return x**3 + 4.7*x**2 + 4.1*x + 0.5

def methodPolovinnogoDeleniya(a , b):
    f_a = f(a)
    for it in range(1000):
        if b - a < 2 * EPS:
            c = 0.5*(a + b)
            f_c = f(c)
            return c,it,f_c
        
        c = 0.5*(a+b)
        f_c = f(c) 
        
        if abs(f_c) < EPS:
            return c,it,f_c
        
        if f_a*f_c < 0:
            b = c
        else:
            a = c
            f_a = f_c 

def methodSecuschih(x0 , x1):
    for it in range(1000):
        f_x0 = f(x0)
        f_x1 = f(x1)

        x2 = x1 - f_x1*(x1 - x0)/(f_x1 - f_x0)
        if abs(x2-x1)<EPS or abs(f(x2)) < EPS:
            return x2,it,f(x2)
        else:
            x0 = x1
            x1 = x2

x = np.linspace(-4,0.5,1000)

plt.figure()
plt.plot(x,f(x),label='$f(x) = x^3 + 4.7x^2 + 4.1x + 0.5$',color='black') 
plt.grid()
plt.show()

a = float(input("Введите левый конец отрезка a: "))
b = float(input("Введите правый конец отрезка b: "))

c,it,f_c = methodPolovinnogoDeleniya(a , b)
c1,it1,f_c1 = methodSecuschih(a , b)

print(f'корни уравнения по методам ПД и секущих:{c},{c1}')
print(f'число итераций у 1. ПД и 2. секущих:{it},{it1}')
print(f'значения в точке f(c) и f(c1): {f_c},{f_c1}')