import sympy as sp
import numpy as np

x , y = 0 , 0
A = np.array([[40,30],[1,1]])
X = np.array([[x],[y]])
B = np.array([[70*0.73],[0.72*2]])

X = np.linalg.solve(A,B)

print(X)