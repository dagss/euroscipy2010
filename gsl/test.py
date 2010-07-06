import numpy as np
import spline
import matplotlib.pyplot as plt

x = np.arange(20)
y = np.array([1, 1.2, 3, 1.2, 0.5, 0, -1, 2, 1, 1, 1, 1.2, 3, 1.2, 0.5, 0, -1, 2, 1, 1])

xhi = np.linspace(0, 20, 200)
xhi = xhi.reshape((2, 10, 10))
yhi = s.evaluate(xhi)
plt.plot(xhi.ravel(), yhi.ravel(), 'b-')
plt.plot(x, y, 'ro')
plt.show()
