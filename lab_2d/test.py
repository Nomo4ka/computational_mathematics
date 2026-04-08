import numpy as np
import matplotlib.pyplot as plt

arr = np.array([i for i in range(5)])
x = np.zeros((5, 5))

x[:,0] = arr
print(x)