import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Known data points
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 1, 4, 9, 16, 25])

# Create interpolation function (linear by default)
f_linear = interp1d(x, y)

# Create interpolation function (cubic)
f_cubic = interp1d(x, y, kind='cubic')

# New x values for which we want interpolated y values
x_new = np.linspace(0, 5, 100)

# Interpolated y values
y_linear = f_linear(x_new)
y_cubic = f_cubic(x_new)

# Plotting the results
plt.plot(x, y, 'o', label='Data points')  # Known data points
plt.plot(x_new, y_linear, '-', label='Linear interpolation')
plt.plot(x_new, y_cubic, '--', label='Cubic interpolation')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interpolation using interp1d')
plt.show()
