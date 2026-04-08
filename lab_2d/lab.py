import numpy as np
import matplotlib.pyplot as plt

a, b = 1.0, 3.0
h = 0.2

def f(x):
    return x / (10 * np.pi * np.sin(x))

def make_table(a, b, h):
    xs = np.arange(a, b + h / 2, h)
    ys = f(xs)
    return xs, ys

def print_table(xs, ys):
    print(" i        x_i          y_i")
    for i in range(len(xs)):
        print(f"{i:2d}   {xs[i]:10.6f}   {ys[i]:10.6f}")

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
    n = len(xs)
    dd = np.zeros((n, n))
    dd[:, 0] = ys

    for j in range(1, n):
        for i in range(n - j):
            dd[i, j] = (dd[i + 1, j - 1] - dd[i, j - 1]) / (xs[i + j] - xs[i])

    return dd

def newton(x, xs, dd):
    n = len(xs)
    s = dd[0, 0]
    p = 1.0
    for j in range(1, n):
        p *= (x - xs[j - 1])
        s += dd[0, j] * p
    return s

def choose_4_nodes(x, xs, ys):
    n = len(xs)

    if x <= xs[1]:
        return xs[:4], ys[:4]
    elif x >= xs[-2]:
        return xs[-4:], ys[-4:]
    else:
        idx = np.searchsorted(xs, x) - 1
        start = max(0, idx - 1)
        if start + 4 > n:
            start = n - 4
        return xs[start:start + 4], ys[start:start + 4]

def local_lagrange(x, xs, ys):
    x4, y4 = choose_4_nodes(x, xs, ys)
    return lagrange(x, x4, y4)

def local_newton(x, xs, ys):
    x4, y4 = choose_4_nodes(x, xs, ys)
    dd = divided_differences(x4, y4)
    return newton(x, x4, dd)

def build_parabolic_spline(xs, ys):
    n = len(xs) - 1  
    h = np.diff(xs)

    a_coef = np.zeros(n)
    b_coef = np.zeros(n)
    c_coef = np.zeros(n)

    c_coef[0] = 0.0
    a_coef[0] = ys[0]
    b_coef[0] = (ys[1] - ys[0] - c_coef[0] * h[0] ** 2) / h[0]

    for i in range(1, n):
        a_coef[i] = ys[i]

        b_coef[i] = b_coef[i - 1] + 2 * c_coef[i - 1] * h[i - 1]

        c_coef[i] = (ys[i + 1] - ys[i] - b_coef[i] * h[i]) / (h[i] ** 2)

    return a_coef, b_coef, c_coef

def main():
    xs, ys = make_table(a, b, h)
    print_table(xs, ys)

    x_test = 1.73
    y_true = f(x_test)
    y_lag = local_lagrange(x_test, xs, ys)
    y_new = local_newton(x_test, xs, ys)

    print("\nПроверка в точке x =", x_test)
    print(f"f(x)          = {y_true:.10f}")
    print(f"Lagrange(x)   = {y_lag:.10f}")
    print(f"Newton(x)     = {y_new:.10f}")
    print(f"|f-L|         = {abs(y_true - y_lag):.10e}")
    print(f"|f-N|         = {abs(y_true - y_new):.10e}")

    x_plot = np.linspace(a, b, 400)
    y_true_plot = f(x_plot)
    y_lag_plot = np.array([local_lagrange(x, xs, ys) for x in x_plot])
    y_new_plot = np.array([local_newton(x, xs, ys) for x in x_plot])

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_true_plot, label='f(x)')
    plt.plot(x_plot, y_lag_plot, label='Lagrange')
    plt.plot(x_plot, y_new_plot, label='Newton')
    plt.scatter(xs, ys, label='Узлы')
    plt.grid(True)
    plt.legend()
    plt.title('Интерполяция функции')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, np.abs(y_true_plot - y_lag_plot), label='|f - Lagrange|')
    plt.plot(x_plot, np.abs(y_true_plot - y_new_plot), label='|f - Newton|')
    plt.grid(True)
    plt.legend()
    plt.title('Абсолютные погрешности')
    plt.xlabel('x')
    plt.ylabel('Ошибка')
    plt.show()

if __name__ == "__main__":
    main()