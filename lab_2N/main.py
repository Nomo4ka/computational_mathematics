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
        self.dd = self._build_divided_differences()

    def _build_divided_differences(self):
        n = len(self.xs)
        dd = np.zeros((n, n))
        dd[:, 0] = self.ys

        for j in range(1, n):
            for i in range(n - j):
                dd[i, j] = (dd[i + 1, j - 1] - dd[i, j - 1]) / (self.xs[i + j] - self.xs[i])

        return dd

    def value(self, x):
        s = self.dd[0, 0]
        p = 1.0
        for j in range(1, len(self.xs)):
            p *= (x - self.xs[j - 1])
            s += self.dd[0, j] * p
        return s

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
        self.M = self.build()

    def build(self):
        n = len(self.xs) - 1
        h = np.diff(self.xs)

        A = np.zeros((n + 1, n + 1))
        rhs = np.zeros(n + 1)

        A[0, 0] = 1.0
        A[n, n] = 1.0

        for i in range(1, n):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            rhs[i] = 6 * (
                (self.ys[i + 1] - self.ys[i]) / h[i]
                - (self.ys[i] - self.ys[i - 1]) / h[i - 1]
            )

        return np.linalg.solve(A, rhs)

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