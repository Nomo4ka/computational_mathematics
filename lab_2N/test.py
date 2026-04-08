import sympy as sp
import numpy as np
import math


EPS = 1e-4
n =3
a , b = 1, 3
x = sp.symbols('x')

f_expr = x / (10 * sp.pi * sp.sin(x))
f4_expr = sp.diff(f_expr, x, 4)
f_der_expr = sp.diff(f_expr, x, n + 1)
f_der_num = sp.lambdify(x, f_der_expr, 'numpy')

print("f(x) =", f_expr)
print("f''''(x) =", sp.simplify(f4_expr))

f_num = sp.lambdify(x, f_expr, 'numpy')
f4_num = sp.lambdify(x, f4_expr, 'numpy')

xs = np.linspace(1, 3, 10000)

f_vals = f_num(xs)
f4_vals = f4_num(xs)
f_der_vals = f_der_num(xs)

max_f = np.max(f_vals)
x_max_f = xs[np.argmax(f_vals)]
max_f4_abs = np.max(np.abs(f4_vals))
x_max_f4_abs = xs[np.argmax(np.abs(f4_vals))]
M_n1 = np.max(np.abs(f_der_vals))
x_M_n1 = xs[np.argmax(np.abs(f_der_vals))]
h_max = ((EPS * math.factorial(n + 1)) / M_n1) ** (1 / (n + 1))

print("\nНа отрезке [1, 3]:")
print(f"Максимум f(x) = {max_f}")
print(f"Достигается при x = {x_max_f}")
print(f"Максимум |f''''(x)| = {max_f4_abs}")
print(f"Достигается при x = {x_max_f4_abs}")
print(h_max)