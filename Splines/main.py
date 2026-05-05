import numpy as np
import matplotlib.pyplot as plt
from math import factorial, sin, cos, pi
from scipy.optimize import minimize_scalar


# Заданная функция
def f(x):
    return x / (10 * np.pi * np.sin(x))


# 4-я производная из расчета
def f4(x):
    ch = cos(x)
    sh = sin(x)

    return (x * (5 + 28 * ch**2 / sh**2 + 24 * ch**4 / sh**4)
            - 4 * (5 + 6 * ch**2 / sh**2) * ch / sh) / (10 * pi * sh)


a = 1
b = 3
eps = 5e-6
degree = 3


# Считаем оценку шага
res = minimize_scalar(lambda t: -abs(f4(t)), bounds=(a, b), method="bounded")
M4 = max(abs(f4(a)), abs(f4(b)), abs(f4(res.x)))

h_est = (factorial(degree + 1) * eps / M4) ** (1 / (degree + 1))

print("M4 =", M4)
print("Оценка шага h =", h_est)

h = 0.01
print("Используем h =", h)


# Узлы таблицы
x_nodes = np.arange(a, b + h / 2, h)
y_nodes = f(x_nodes)


# Интерполяция Лагранжа по 4 ближайшим точкам
def lagrange(x, xs, ys):
    ans = 0

    for i in range(4):
        p = ys[i]

        for j in range(4):
            if i != j:
                p *= (x - xs[j]) / (xs[i] - xs[j])

        ans += p

    return ans


def take_4_nodes(x):
    i = np.searchsorted(x_nodes, x) - 1

    if i < 1:
        return [0, 1, 2, 3]

    if i > len(x_nodes) - 3:
        m = len(x_nodes) - 1
        return [m - 3, m - 2, m - 1, m]

    return [i - 1, i, i + 1, i + 2]


def lagrange_local(x):
    ids = take_4_nodes(x)
    return lagrange(x, x_nodes[ids], y_nodes[ids])


# Ньютон через конечные разности
def make_differences(ys):
    dy = [ys.copy().astype(float)]

    for i in range(1, len(ys)):
        dy.append(np.diff(dy[i - 1]))

    return dy


def newton(x, xs, ys):
    dy = make_differences(ys)
    step = xs[1] - xs[0]
    t = (x - xs[0]) / step

    ans = dy[0][0]
    mult = 1

    for k in range(1, len(dy)):
        mult *= (t - k + 1) / k
        ans += mult * dy[k][0]

    return ans


def newton_local(x):
    ids = take_4_nodes(x)
    return newton(x, x_nodes[ids], y_nodes[ids])


# Параболический сплайн
# На каждом отрезке: S = A + B*(x-xi) + C*(x-xi)^2
def make_parabola_spline(x, y):
    count = len(x) - 1
    step = np.diff(x)

    A = y[:-1].copy()
    B = np.zeros(count)
    C = np.zeros(count)

    # Дополнительное условие: S''(a) = 0
    C[0] = 0
    B[0] = (y[1] - y[0]) / step[0]

    for i in range(1, count):
        B[i] = B[i - 1] + 2 * C[i - 1] * step[i - 1]
        C[i] = (y[i + 1] - y[i] - B[i] * step[i]) / step[i]**2

    return A, B, C


def parabola_value(x, nodes, coef):
    A, B, C = coef

    i = np.searchsorted(nodes, x) - 1
    i = max(0, min(i, len(A) - 1))

    dx = x - nodes[i]
    return A[i] + B[i] * dx + C[i] * dx**2


# Кубический сплайн
# На каждом отрезке: S = A + B*(x-xi) + C*(x-xi)^2 + D*(x-xi)^3
def make_cubic_spline(x, y):
    count = len(x) - 1
    step = np.diff(x)

    A = y[:-1].copy()

    matrix = np.zeros((count + 1, count + 1))
    right = np.zeros(count + 1)

    # Дополнительные условия: S''(a) = 0 и S''(b) = 0
    matrix[0, 0] = 1
    matrix[count, count] = 1

    for i in range(1, count):
        matrix[i, i - 1] = step[i - 1]
        matrix[i, i] = 2 * (step[i - 1] + step[i])
        matrix[i, i + 1] = step[i]

        left = (y[i] - y[i - 1]) / step[i - 1]
        right_slope = (y[i + 1] - y[i]) / step[i]
        right[i] = 3 * (right_slope - left)

    C_all = np.linalg.solve(matrix, right)

    B = np.zeros(count)
    C = C_all[:-1]
    D = np.zeros(count)

    for i in range(count):
        B[i] = (y[i + 1] - y[i]) / step[i] - step[i] * (2 * C_all[i] + C_all[i + 1]) / 3
        D[i] = (C_all[i + 1] - C_all[i]) / (3 * step[i])

    return A, B, C, D


def cubic_value(x, nodes, coef):
    A, B, C, D = coef

    i = np.searchsorted(nodes, x) - 1
    i = max(0, min(i, len(A) - 1))

    dx = x - nodes[i]
    return A[i] + B[i] * dx + C[i] * dx**2 + D[i] * dx**3


parabola_coef = make_parabola_spline(x_nodes, y_nodes)
cubic_coef = make_cubic_spline(x_nodes, y_nodes)


# Проверяем на частой сетке
x_test = np.linspace(a, b, 5000)
y_true = f(x_test)

y_lagrange = np.array([lagrange_local(x) for x in x_test])
y_newton = np.array([newton_local(x) for x in x_test])
y_parabola = np.array([parabola_value(x, x_nodes, parabola_coef) for x in x_test])
y_cubic = np.array([cubic_value(x, x_nodes, cubic_coef) for x in x_test])

err_lagrange = abs(y_true - y_lagrange)
err_newton = abs(y_true - y_newton)
err_parabola = abs(y_true - y_parabola)
err_cubic = abs(y_true - y_cubic)

print("\nОшибки:")
print("Лагранж:", max(err_lagrange))
print("Ньютон:", max(err_newton))
print("Параболический сплайн:", max(err_parabola))
print("Кубический сплайн:", max(err_cubic))


# Графики интерполяции
plt.figure()
plt.plot(x_test, y_true, label="f(x)")
plt.plot(x_test, y_lagrange, "--", label="Лагранж")
plt.plot(x_test, y_newton, ":", label="Ньютон")
plt.plot(x_test, y_parabola, "-.", label="Параболический сплайн")
plt.plot(x_test, y_cubic, "--", label="Кубический сплайн")
plt.legend()
plt.grid()
plt.title("Интерполяция")
plt.show()


# Графики ошибок
plt.figure()
plt.plot(x_test, err_lagrange, label="Лагранж")
plt.plot(x_test, err_newton, label="Ньютон")
plt.plot(x_test, err_parabola, label="Параболический")
plt.plot(x_test, err_cubic, label="Кубический")
plt.legend()
plt.grid()
plt.title("Ошибки")
plt.show()
