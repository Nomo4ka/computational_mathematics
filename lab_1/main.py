import numpy as np
import matplotlib.pyplot as plt

#Вариант 7

EPS =1e-3

class Result:
    def __init__(self, x, it, f_x):
        self.x = x  
        self.it = it  
        self.f_x = f_x  
    
    def display(self):
        print(f'корень уравнения: {self.x}')
        print(f'число итераций: {self.it}')
        print(f"невязка: {abs(self.f_x)}")

def f(x):
    return x**3 + 4.7*x**2 + 4.1*x + 0.5

def df_x(x):
    return 3*x**2 + 9.4*x + 4.1

def methodPolovinnogoDeleniya(a:float , b:float):
    f_a = f(a)
    it = 0
    while abs(b - a) >=  2 * EPS:
        c = 0.5*(a + b)
        f_c = f(c)
        it += 1
        
        if abs(f_c) < EPS:
            return Result(c, it, f_c)
        
        if f_a*f_c < 0:
            b = c
        else:
            a = c
            f_a = f_c 

    c = 0.5*(a + b)
    f_c = f(c)
    return Result(c, it, f_c)

def methodSecuschih(x0:float):
    x1 = x0 - f(x0)/df_x(x0)
    f_x0 = f(x0)
    f_x1 = f(x1)
    it = 0
        
    while abs(x1 - x0) > EPS:

        x2 = x1 - f_x1*(x1 - x0)/(f_x1 - f_x0)
        f_x2 = f(x2)
        it+=1
            
        x0 = x1
        x1 = x2
        f_x0 = f_x1
        f_x1 = f_x2
    return Result(x1, it, f_x1)

def graphics():
    x = np.linspace(-4,0.5,1000)
    plt.figure()
    plt.plot(x,f(x),label='$f(x) = x^3 + 4.7x^2 + 4.1x + 0.5$',color='black') 
    plt.grid()
    plt.show()

def userInput():
    a = float(input("Левая граница a = "))
    b = float(input("Правая граница b = "))
    x0 = float(input("\nВведите начальное приближение: "))
    
    return a, b, x0

def main():
    graphics()
    a , b ,x0 = userInput()

    print('\nМЕТОД ПОЛОВИННОГО ДЕЛЕНИЯ')
    result = methodPolovinnogoDeleniya(a , b)
    result.display()

    print('\nМЕТОД СЕКУЩИХ')
    result1  = methodSecuschih(x0)
    result1.display()
            
if __name__ == "__main__":
    main()