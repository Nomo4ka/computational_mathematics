#Вариант 7
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

EPS = 1e-5

class Result:
    def __init__(self, x, y,it ,f_x):
        self.x = x  
        self.y = y
        self.it = it  
        self.f_x = f_x  
    
    def display(self):
        print(f'корень уравнения: {self.x} , {self.y}')
        print(f'число итераций: {self.it}')

def Jacobian(X):
    J = np.array([[np.cos(X[0]), 2],
                  [2, -np.sin(X[1] - 1)]])
    return J

def vecFx(X):
    f1 = np.sin(X[0]) + 2*X[1] - 2
    f2 = 2*X[0] + np.cos(X[1]- 1) - 0.7
    return np.array([f1,f2])

def Newton_method(X0):
    J = Jacobian(X0)
    X = np.array(X0)
    vall = vecFx(X0)
    Xpp = X - np.dot(np.linalg.inv(J),vall)
    it = 0
    
    while np.linalg.norm(Xpp - X) >= EPS:
        X = Xpp
        J = Jacobian(X)
        X = np.array(X)
        vall = vecFx(X)
        Xpp = X - np.dot(np.linalg.inv(J),vall) 
        it +=1

    return Result(Xpp[0], Xpp[1], it, vecFx(Xpp))

def userInp():
    x = float(input("введите начальное приближение по х: "))
    y = float(input("введите начальное приближение по у: "))
    
    return np.array([x,y])
    
def graphics():
    x, y = sp.symbols('x y')
    
    f = sp.sin(x) + 2*y - 2
    g = 2*x + sp.cos(y - 1) - 0.7
    
    f = sp.lambdify((x, y), f, 'numpy')
    g = sp.lambdify((x, y), g, 'numpy')
    
    xVals = np.linspace(-10, 10, 400)
    yVals = np.linspace(-10, 10, 400)
    X, Y = np.meshgrid(xVals, yVals)
    
    Z1 = f(X, Y)
    Z2 = g(X, Y)
    
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z1, levels=[0], colors='blue', )
    plt.contour(X, Y, Z2, levels=[0])
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.show()
    
def main():
    graphics()
    
if __name__ == "__main__":
    main() 