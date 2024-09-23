import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import numpy as np
from pyquaternion import Quaternion


def interpolate():
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


def global_to_ego(global_position, ego_translation, ego_rotation):
    """
    Convert global coordinates to ego vehicle coordinates.
    
    Parameters:
    - global_position: 3D position in global coordinates (x, y, z)
    - ego_translation: 3D translation of the ego vehicle in global coordinates (x, y, z)
    - ego_rotation: Quaternion representing the ego vehicle's rotation (qx, qy, qz, qw)
    
    Returns:
    - Position in ego coordinates (x, y, z)
    """
    # Convert ego rotation (quaternion) to a rotation matrix
    q_ego = Quaternion(ego_rotation)
    
    # Invert the ego rotation (take the conjugate of the quaternion)
    q_ego_inv = q_ego.inverse
    
    # Subtract ego translation from the global position
    relative_position = np.array(global_position) - np.array(ego_translation)
    
    # Apply the inverse rotation to the relative position
    ego_position = q_ego_inv.rotate(relative_position)
    
    return ego_position

# Example usage:
global_position = [100, 200, 0]  # Example global coordinates
ego_translation = [90, 195, 0]   # Ego position in global coordinates
ego_rotation = [0, 0, 0.7071, 0.7071]  # Example ego rotation (quaternion format)

ego_coords = global_to_ego(global_position, ego_translation, ego_rotation)
print(f"Position in ego coordinates: {ego_coords}")