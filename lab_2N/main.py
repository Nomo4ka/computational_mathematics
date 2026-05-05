import numpy as np
import matplotlib.pyplot as plt

class FunctionData:
    def __init__(self, xs):
        self.xs = np.array(xs, dtype=float)
        self.ys = self.f(self.xs)

    def f(self, x):
        return x / (10 * np.pi * np.sin(x))

class NewtonInterpolator:
    def __init__(self, xs, ys):
        self.xs = np.array(xs, dtype=float)
        self.ys = np.array(ys, dtype=float)
        self.h = self.xs[1] - self.xs[0]
        self.diff_table = self.build_table()

    def build_table(self):
        n = len(self.ys)
        table = np.zeros((n, n))
        table[:, 0] = self.ys

        for j in range(1, n):
            for i in range(n - j):
                table[i][j] = table[i + 1][j - 1] - table[i][j - 1]

        return table

    def value(self, x):
        t = (x - self.xs[0]) / self.h

        result = self.diff_table[0][0]

        fact = 1
        t_term = 1

        for i in range(1, len(self.xs)):
            t_term *= (t - (i - 1))
            fact *= i
            result += (t_term / fact) * self.diff_table[0][i]

        return result

class LagrangeInterpolator:
    def __init__(self, xs, ys):
        self.xs = np.array(xs, dtype=float)
        self.ys = np.array(ys, dtype=float)

    def value(self, x):
        n = len(self.xs)
        s = 0.0
        for i in range(n):
            p = 1.0
            for j in range(n):
                if i != j:
                    p *= (x - self.xs[j]) / (self.xs[i] - self.xs[j])
            s += self.ys[i] * p
        return s

class ParabolicSpline:
    def __init__(self, xs, ys):
        self.xs = np.array(xs, dtype=float)
        self.ys = np.array(ys, dtype=float)
        self.a, self.b, self.c = self.build()

    def build(self):
        n = len(self.xs) - 1
        h = np.diff(self.xs)

        a = np.zeros(n)
        b = np.zeros(n)
        c = np.zeros(n)

        a[0] = self.ys[0]
        c[0] = 0.0
        b[0] = (self.ys[1] - self.ys[0]) / h[0]

        for i in range(1, n):
            a[i] = self.ys[i]
            b[i] = b[i - 1] + 2 * c[i - 1] * h[i - 1]
            c[i] = (self.ys[i + 1] - self.ys[i] - b[i] * h[i]) / (h[i] ** 2)

        return a, b, c

    def value(self, x):
        i = np.searchsorted(self.xs, x) - 1
        if i < 0:
            i = 0
        if i >= len(self.xs) - 1:
            i = len(self.xs) - 2

        dx = x - self.xs[i]
        return self.a[i] + self.b[i] * dx + self.c[i] * dx ** 2


class CubicSpline:
    def __init__(self, xs, ys):
        self.xs = np.array(xs, dtype=float)
        self.ys = np.array(ys, dtype=float)
        self.a, self.b, self.c, self.d = self.build()

    def thomas_method(self, lower, diag, upper, rhs):
        n = len(diag)

        C = np.zeros(n)
        D = np.zeros(n)

        C[0] = upper[0] / diag[0]
        D[0] = rhs[0] / diag[0]

        for i in range(1, n):
            denom = diag[i] - lower[i] * C[i - 1]
            if i < n - 1:
                C[i] = upper[i] / denom
            D[i] = (rhs[i] - lower[i] * D[i - 1]) / denom

        x = np.zeros(n)
        x[-1] = D[-1]

        for i in range(n - 2, -1, -1):
            x[i] = D[i] - C[i] * x[i + 1]

        return x

    def build(self):
        n = len(self.xs) - 1
        h = np.diff(self.xs)

        lower = np.zeros(n + 1)
        diag = np.zeros(n + 1)
        upper = np.zeros(n + 1)
        rhs = np.zeros(n + 1)

        diag[0] = 1.0
        rhs[0] = 0.0

        diag[n] = 1.0
        rhs[n] = 0.0

        for i in range(1, n):
            lower[i] = h[i - 1]
            diag[i] = 2 * (h[i - 1] + h[i])
            upper[i] = h[i]
            rhs[i] = 3 * (
                (self.ys[i + 1] - self.ys[i]) / h[i]
                - (self.ys[i] - self.ys[i - 1]) / h[i - 1]
            )

        c = self.thomas_method(lower, diag, upper, rhs)

        a = np.zeros(n)
        b = np.zeros(n)
        d = np.zeros(n)

        for i in range(n):
            a[i] = self.ys[i]
            b[i] = (self.ys[i + 1] - self.ys[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3
            d[i] = (c[i + 1] - c[i]) / (3 * h[i])

        return a, b, c, d

    def value(self, x):
        n = len(self.xs) - 1

        i = np.searchsorted(self.xs, x) - 1
        if i < 0:
            i = 0
        if i >= n:
            i = n - 1

        dx = x - self.xs[i]

        return (
            self.a[i]
            + self.b[i] * dx
            + self.c[i] * dx ** 2
            + self.d[i] * dx ** 3
        )

    def value(self, x):
        i = np.searchsorted(self.xs, x) - 1
        if i < 0:
            i = 0
        if i >= len(self.xs) - 1:
            i = len(self.xs) - 2

        h = self.xs[i + 1] - self.xs[i]

        return (
            self.M[i] * (self.xs[i + 1] - x) ** 3 / (6 * h)
            + self.M[i + 1] * (x - self.xs[i]) ** 3 / (6 * h)
            + (self.ys[i] - self.M[i] * h ** 2 / 6) * (self.xs[i + 1] - x) / h
            + (self.ys[i + 1] - self.M[i + 1] * h ** 2 / 6) * (x - self.xs[i]) / h
        )

class Lab:
    def __init__(self, xs, x_star):
        self.data = FunctionData(xs)
        self.x_star = x_star

        self.lagrange = LagrangeInterpolator(self.data.xs, self.data.ys)
        self.newton = NewtonInterpolator(self.data.xs, self.data.ys)
        self.parabolic = ParabolicSpline(self.data.xs, self.data.ys)
        self.cubic = CubicSpline(self.data.xs, self.data.ys)

    def print_nodes(self):
        for i in range(len(self.data.xs)):
            print(f"x[{i}] = {self.data.xs[i]:.6f}, y[{i}] = {self.data.ys[i]:.10f}")

    def compute(self):
        y_true = self.data.f(self.x_star)
        y_lag = self.lagrange.value(self.x_star)
        y_new = self.newton.value(self.x_star)
        y_par = self.parabolic.value(self.x_star)
        y_cub = self.cubic.value(self.x_star)

        print(f"\nТочка интерполирования x* = {self.x_star:.6f}")
        print(f"Точное значение f(x*) = {y_true:.10f}\n")

        print(f"Lagrange      = {y_lag:.10f}")
        print(f"Newton        = {y_new:.10f}")
        print(f"Parabolic     = {y_par:.10f}")
        print(f"Cubic spline  = {y_cub:.10f}\n")

        print(f"|f-L| = {abs(y_true - y_lag):.10e}")
        print(f"|f-N| = {abs(y_true - y_new):.10e}")
        print(f"|f-P| = {abs(y_true - y_par):.10e}")
        print(f"|f-C| = {abs(y_true - y_cub):.10e}")

    def plot(self):
        x_plot = np.linspace(self.data.xs[0], self.data.xs[-1], 300)
        y_true = self.data.f(x_plot)

        y_lag = np.array([self.lagrange.value(x) for x in x_plot])
        y_new = np.array([self.newton.value(x) for x in x_plot])
        y_par = np.array([self.parabolic.value(x) for x in x_plot])
        y_cub = np.array([self.cubic.value(x) for x in x_plot])

        plt.figure(figsize=(10, 6))
        plt.plot(x_plot, y_true, label='f(x)')
        plt.plot(x_plot, y_lag, label='Lagrange')
        plt.plot(x_plot, y_new, label='Newton')
        plt.plot(x_plot, y_par, label='Parabolic spline')
        plt.plot(x_plot, y_cub, label='Cubic spline')
        plt.scatter(self.data.xs, self.data.ys, label='Узлы')
        plt.grid(True)
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(x_plot, np.abs(y_true - y_lag), label='|f - Lagrange|')
        plt.plot(x_plot, np.abs(y_true - y_new), label='|f - Newton|')
        plt.plot(x_plot, np.abs(y_true - y_par), label='|f - Parabolic|')
        plt.plot(x_plot, np.abs(y_true - y_cub), label='|f - Cubic|')
        plt.grid(True)
        plt.legend()
        plt.title('Абсолютные погрешности')
        plt.xlabel('x')
        plt.ylabel('Ошибка')
        plt.show()

def main():
    xs = [1.72, 1.73, 1.74, 1.75]
    x_star = 1.73

    lab = Lab(xs, x_star)
    lab.print_nodes()
    lab.compute()
    lab.plot()

if __name__ == "__main__":
    main()