import numpy as np
import matplotlib.pyplot as plt

a, b = 1.0, 3.0

def f(x):
    return x / (10 * np.pi * np.sin(x))

x_nodes = np.linspace(a, b, 4)
y_nodes = f(x_nodes)

def lagrange(x, xs, ys):
    n = len(xs)
    s = 0.0
    for i in range(n):
        p = 1.0
        for j in range(n):
            if i != j:
                p *= (x - xs[j]) / (xs[i] - xs[j])
        s += ys[i] * p
    return s


def divided_differences(xs, ys):
    coef = ys.astype(float).copy()
    n = len(xs)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j - 1:n - 1]) / (xs[j:n] - xs[:n - j])
    return coef

def newton(x, xs, coef):
    p = coef[-1]
    for k in range(len(coef) - 2, -1, -1):
        p = p * (x - xs[k]) + coef[k]
    return p

newton_coef = divided_differences(x_nodes, y_nodes)

def build_parabolic_spline(xs, ys):
    n = len(xs) - 1
    h = xs[1:] - xs[:-1]

    a_coef = ys[:-1].copy()
    b_coef = np.zeros(n)
    c_coef = np.zeros(n)

    c_coef[0] = 0.0
    b_coef[0] = (ys[1] - ys[0] - c_coef[0] * h[0] ** 2) / h[0]

    for i in range(n - 1):
        c_coef[i] = (ys[i + 1] - ys[i] - b_coef[i] * h[i]) / h[i] ** 2
        b_coef[i + 1] = b_coef[i] + 2 * c_coef[i] * h[i]

    c_coef[-1] = (ys[-1] - ys[-2] - b_coef[-1] * h[-1]) / h[-1] ** 2
    return a_coef, b_coef, c_coef

def parabolic_spline(x, xs, a_coef, b_coef, c_coef):
    if x <= xs[0]:
        i = 0
    elif x >= xs[-1]:
        i = len(xs) - 2
    else:
        i = np.searchsorted(xs, x) - 1

    dx = x - xs[i]
    return a_coef[i] + b_coef[i] * dx + c_coef[i] * dx ** 2

par_a, par_b, par_c = build_parabolic_spline(x_nodes, y_nodes)

def build_cubic_spline(xs, ys):
    n = len(xs) - 1
    h = xs[1:] - xs[:-1]

    alpha = np.zeros(n + 1)
    for i in range(1, n):
        alpha[i] = 3 / h[i] * (ys[i + 1] - ys[i]) - 3 / h[i - 1] * (ys[i] - ys[i - 1])

    l = np.ones(n + 1)
    mu = np.zeros(n + 1)
    z = np.zeros(n + 1)

    for i in range(1, n):
        l[i] = 2 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    a_coef = ys[:-1].copy()
    b_coef = np.zeros(n)
    c_coef = np.zeros(n + 1)
    d_coef = np.zeros(n)

    for j in range(n - 1, -1, -1):
        c_coef[j] = z[j] - mu[j] * c_coef[j + 1]
        b_coef[j] = (ys[j + 1] - ys[j]) / h[j] - h[j] * (c_coef[j + 1] + 2 * c_coef[j]) / 3
        d_coef[j] = (c_coef[j + 1] - c_coef[j]) / (3 * h[j])

    return a_coef, b_coef, c_coef[:-1], d_coef

def cubic_spline(x, xs, a_coef, b_coef, c_coef, d_coef):
    if x <= xs[0]:
        i = 0
    elif x >= xs[-1]:
        i = len(xs) - 2
    else:
        i = np.searchsorted(xs, x) - 1

    dx = x - xs[i]
    return a_coef[i] + b_coef[i] * dx + c_coef[i] * dx ** 2 + d_coef[i] * dx ** 3

cub_a, cub_b, cub_c, cub_d = build_cubic_spline(x_nodes, y_nodes)


x_dense = np.linspace(a, b, 500)
y_true = f(x_dense)

y_lagr = np.array([lagrange(x, x_nodes, y_nodes) for x in x_dense])
y_newt = np.array([newton(x, x_nodes, newton_coef) for x in x_dense])
y_par  = np.array([parabolic_spline(x, x_nodes, par_a, par_b, par_c) for x in x_dense])
y_cub  = np.array([cubic_spline(x, x_nodes, cub_a, cub_b, cub_c, cub_d) for x in x_dense])

err_lagr = np.abs(y_true - y_lagr)
err_newt = np.abs(y_true - y_newt)
err_par  = np.abs(y_true - y_par)
err_cub  = np.abs(y_true - y_cub)


print("Узлы интерполяции:")
print(" i        x_i           f(x_i)")
for i in range(len(x_nodes)):
    print(f"{i:2d}   {x_nodes[i]:10.6f}   {y_nodes[i]:14.10f}")

print("\nМаксимальная абсолютная ошибка:")
print(f"Лагранж:              {np.max(err_lagr):.10e}")
print(f"Ньютон:               {np.max(err_newt):.10e}")
print(f"Параболический сплайн:{np.max(err_par):.10e}")
print(f"Кубический сплайн:    {np.max(err_cub):.10e}")

plt.figure()
plt.plot(x_dense, y_true, label="f(x)")
plt.plot(x_dense, y_lagr, label="Лагранж")
plt.plot(x_dense, y_newt, label="Ньютон")
plt.plot(x_dense, y_par, label="Параболический сплайн")
plt.plot(x_dense, y_cub, label="Кубический сплайн")

plt.scatter(x_nodes, y_nodes)  

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()
plt.figure()
plt.plot(x_dense, err_lagr, label="Лагранж")
plt.plot(x_dense, err_newt, label="Ньютон")
plt.plot(x_dense, err_par, label="Параболический")
plt.plot(x_dense, err_cub, label="Кубический")

plt.xlabel("x")
plt.ylabel("Ошибка")
plt.legend()
plt.grid()

plt.show()