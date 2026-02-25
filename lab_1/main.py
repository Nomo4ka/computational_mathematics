import math
import numpy as np
import matplotlib.pyplot as plt

#Вариант 7

EPS =1e-3

def f(x):
    return x**3 + 4.7*x**2 + 4.1*x + 0.5

def methodPolovinnogoDeleniya(a , b):
    f_a = f(a)
    it = 0
    while abs(b - a) >=  2 * EPS:
        c = 0.5*(a + b)
        f_c = f(c)
        it += 1
        
        if abs(f_c) < EPS:
            return c,it,f_c
        
        if f_a*f_c < 0:
            b = c
        else:
            a = c
            f_a = f_c 

    c = 0.5*(a + b)
    f_c = f(c)
    return c, it, f_c

def methodSecuschih(x0 , x1):
    f_x0 = f(x0)
    f_x1 = f(x1)
    it = 0
        
    while abs(x1 - x0) > EPS and abs(f_x1) > EPS:

        x2 = x1 - f_x1*(x1 - x0)/(f_x1 - f_x0)
        f_x2 = f(x2)
        it+=1
            
        x0 = x1
        x1 = x2
        f_x0 = f_x1
        f_x1 = f_x2
    return x1,it,f_x1
            


def show(x,it,f_x):
    print(f'корень уравнения: {x}')
    print(f'число итераций: {it}')
    print(f"невязка: {abs(f_x)}")
    
x = np.linspace(-4,0.5,1000)

plt.figure()
plt.plot(x,f(x),label='$f(x) = x^3 + 4.7x^2 + 4.1x + 0.5$',color='black') 
plt.grid()
plt.show()

a = float(input("Введите левый конец отрезка a: "))
b = float(input("Введите правый конец отрезка b: "))

while 1:   
    choice = int(input("Выберите метод:\n1.Половинного деления \n2.Секущих\n"))
    match choice:
        case 1:
            print('\nМЕТОД ПОЛОВИННОГО ДЕЛЕНИЯ')
            c,it,f_c = methodPolovinnogoDeleniya(a , b)
            show(c,it,f_c)
        case 2:
            print('\nМЕТОД СЕКУЩИХ')
            c1,it1,f_c1 = methodSecuschih(a , b)
            show(c1,it1,f_c1)
            
    c = input('\nПродолжить? Введите Y или y, чтобы подтвердить , иначе - нет: ')
    if(c != 'Y' and c != 'y'):
        break