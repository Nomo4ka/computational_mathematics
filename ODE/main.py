import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-4

x0 = 1
y0 = 1
b = 2

def f(x, y):
    return (2 * y**2 * np.log(x) - y) / x

def analytical_solution(x):
    return 1 / (2 * np.log(x) + 2 - x)

class Euler:
    def __init__(self, f):
        self.f = f

    def step(self, x, y, h):
        return y + h * self.f(x, y)

    def solve(self, x0, y0, x_end, h):
        xs = [x0]
        ys = [y0]

        x = x0
        y = y0

        while x < x_end:
            h_step = min(h, b - x)
            y = self.step(x, y, h_step)
            x += h_step

            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)

class RungeKutta:
    def __init__(self, f):
        self.f = f

    def step(self, x, y, h):
        k1 = self.f(x, y)
        k2 = self.f(x + h / 2, y + h * k1 / 2)
        k3 = self.f(x + h / 2, y + h * k2 / 2)
        k4 = self.f(x + h, y + h * k3)

        return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def solve(self, x0, y0, x_end, h):
        xs = [x0]
        ys = [y0]

        x = x0
        y = y0

        while x < x_end:
            h_step = min(h, x_end - x)
            y = self.step(x, y, h_step)
            x += h_step

            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)


euler = Euler(f)
rk = RungeKutta(f)

def solution_at_end(method, x0, y0, b, h):
    _, ys = method.solve(x0, y0, b, h)
    return ys[-1]

def step_delta(method, x0, y0, b, h):
    y_h = solution_at_end(method, x0, y0, b, h)
    y_h2 = solution_at_end(method, x0, y0, b, h / 2)
    return abs(y_h2 - y_h) / 15

def findh(x0, y0, b, h_alleged):
    h = h_alleged
    delta = step_delta(rk, x0, y0, b, h)
    
    while delta > EPS:
        h /= 2
        delta = step_delta(rk, x0, y0, b, h)
    
    return h

def div():
    h = findh(x0, y0, b, 0.1)
    
    x_euler, y_euler = euler.solve(x0, y0, b, h)
    x_rk, y_rk = rk.solve(x0, y0, b, h)

    y_analyt = analytical_solution(x_rk)

    delta_euler = np.abs(analytical_solution(x_euler) - y_euler)
    delta_rk = np.abs(y_analyt - y_rk)

    return delta_euler, delta_rk

def plot_solutions():
    h = findh(x0, y0, b, 0.1)

    x_euler, y_euler = euler.solve(x0, y0, b, h)
    x_rk, y_rk = rk.solve(x0, y0, b, h)

    x_dense = np.linspace(x0, b, 500)
    y_exact_dense = analytical_solution(x_dense)

    plt.figure()
    plt.plot(x_dense, y_exact_dense, label="Аналитическое решение", linewidth=2)
    plt.plot(x_euler, y_euler, "o-", label="Эйлер")
    plt.plot(x_rk, y_rk, "s-", label="Рунге-Кутта")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()

    plt.show()

def table(x0, y0, b, h):
    x_euler, y_euler = euler.solve(x0, y0, b, h)
    x_rk, y_rk = rk.solve(x0, y0, b, h)

    x_nodes = x_rk

    y_exact = analytical_solution(x_nodes)

    delta_euler = np.abs(y_exact - y_euler)
    delta_rk = np.abs(y_exact - y_rk)

    print("\Table:")
    print(f"{'x':>8} {'y_exact':>12} {'y_euler':>12} {'y_rk':>12} {'d_euler':>12} {'d_rk':>12}")

    for i in range(len(x_nodes)):
        print(f"{x_nodes[i]:8.4f} "
              f"{y_exact[i]:12.6f} "
              f"{y_euler[i]:12.6f} "
              f"{y_rk[i]:12.6f} "
              f"{delta_euler[i]:12.6f} "
              f"{delta_rk[i]:12.6f}")

    return x_nodes, y_exact, y_euler, y_rk, delta_euler, delta_rk

def main():
    h = findh(x0, y0, b, 0.1)
    
    delta_euler, delta_rk = div()
    print(f"max d(Euler) = {np.max(delta_euler)}")
    print(f"max d(Runge-Kutta) = {np.max(delta_rk)}")
    plot_solutions()
    table(x0, y0, b, h)

if __name__ == "__main__":
    main()
