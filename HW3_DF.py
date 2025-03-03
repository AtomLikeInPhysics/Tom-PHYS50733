#!/usr/bin/env python3

"""
HW3.py

Author: Danette Farnsworth
Date: 2025-02-18
"""


import matplotlib.pyplot as plt
import numpy as np

###################################################################
# Problem 1
# Numerical Derivative VS Known Derivative
###################################################################
# Consider a function 1+(2/2)+ tan(2x). You should be able to write the derivate
# without much effort. Calculate the derivative of this function in the range
# -2 <x<2 unsing central difference method. Plot your computed derivative as points
# and use a line to plot the analytic solution through the same points. How
# accurate is your computed derivative?
###################################################################

print("###########################################################")
print("Derivative in the range -2<x<2 using Central Method")
print("###########################################################")
print("\n")


def F(x):
    return 1.0 + (1.0 / 2.0) + np.tanh(2.0 * x)


def derivative(x):
    """Calulate derivative of F(x)"""
    return 2.0 / np.cosh(2.0 * x) ** 2


def central_difference(f, x, h=0.000001):
    """Central Difference method for numerical derivative"""
    return (f(x + h) - f(x - h)) / (2 * h)


# define x Values
x_vals = np.linspace(-2.0, 2.0, 100)
numerical_derivative = central_difference(F, x_vals)
analytical_derivative = derivative(x_vals)

# Plot Results
plt.figure(figsize=(8, 5))
plt.plot(x_vals, analytical_derivative, label="Analytical Derivative", color="pink", linewidth=2, zorder=1)
plt.scatter(x_vals, numerical_derivative, color="green", label="Numerical Derivative", marker="*", s=10, zorder=2)
plt.xlabel("x")
plt.ylabel("f'(x)")
plt.title("Comparison of Numerical and Analytical Derivatives")
plt.legend()
plt.grid(True)
plt.show()

# Compute error
error = np.abs(numerical_derivative - analytical_derivative)
max_error = np.max(error)
max_error

###################################################################
# Problem 2
# Electric Field of a Charge Distribution
###################################################################
# Consider two charges, of +/-C, 10 cm apart. Calculatej the electric
# potential on a 1mx1m plane surrounding the charges, using a grid of
# points spaced 1cm aprat. Plot the potential.
###################################################################

print("###########################################################")
print("Calculated electric potenial of two charges +/-C 10cm apart")
print("###########################################################")
print("\n")

# Constants
epsO = 8.85e-12  # Permittivity of free space (F/m)
q = 1e-9  # Charge magnitude (Coulombs)
d = 0.1  # Distance between charges (cm)
grid_size = 0.5  # Grid size (cm)
spacing = 0.005  # Grid spacing (cm)
softening = 1e-1  # Softening parameter to avoid singularities

# Create grid points
x = np.arange(-grid_size / 2, grid_size / 2 + spacing, spacing)
y = np.arange(-grid_size / 2, grid_size / 2 + spacing, spacing)
X, Y = np.meshgrid(x, y)

# Position of the charges
pos_q = (-d / 2, 0)  # Positive charge at (-d/2, 0)
neg_q = (d / 2, 0)  # Negative charge at (d / 2, 0)


# Function to compute potential
def potential(q, xq, yq, X, Y):
    r = np.sqrt((X - xq)**2 + (Y - yq)**2) + softening
    return q / (4 * np.pi * epsO * r)


# Compute potential due to both charges
phi_plus = potential(q, *pos_q, X, Y)
phi_minus = potential(-q, *neg_q, X, Y)

# Total potential
phi_total = phi_plus + phi_minus

# Set symmetric color limits for better visualation
vmax = np.max(np.abs(phi_total))  # Define vmax based on the potential range

# Plot the potential
plt.figure(figsize=(8, 6))
plt.imshow(phi_total, extent=[-grid_size / 2, grid_size / 2, -grid_size / 2, grid_size / 2], origin="lower", cmap="coolwarm")
plt.scatter([pos_q[0], neg_q[0]], [pos_q[1], neg_q[1]], color="black", marker="o", label="Charges")
plt.colorbar(label="Electric Potential (V)")
plt.title("Electric Potential of a Dipole")
plt.xlabel("x (cm)")
plt.ylabel("y (cm)")
plt.legend()
plt.show()

# Compute electric field components (negative gradient of potential)
Ey, Ex = np.gradient(-phi_total, spacing)  # Note: np.gradient returns [d/dy, d/dx]

# Compute magnitude of the electric field
E_magnitude = np.sqrt(Ex**2 + Ey**2)

# Limit the magnitude to avoid excessive values near charges
E_max = np.percentile(E_magnitude, 95)  # Set max threshold to 95th percentile
scaling_factor = np.clip(E_max / (E_magnitude + softening), 0, 1)
Ex *= scaling_factor
Ey *= scaling_factor

# Subsample for quiver plot (to avoid too many arrows)
step = 5  # Adjust to control arrow density
X_quiver = X[::step, ::step]
Y_quiver = Y[::step, ::step]
Ex_quiver = Ex[::step, ::step]
Ey_quiver = Ey[::step, ::step]

# Plot the electric field vectors using quiver
plt.figure(figsize=(8, 6))
plt.imshow(phi_total, extent=[-grid_size / 2, grid_size / 2, -grid_size / 2, grid_size / 2],
           origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
plt.colorbar(label="Electric Potential (V)")
plt.quiver(X_quiver, Y_quiver, Ex_quiver, Ey_quiver, color="black")
plt.title("Electric Field of a Dipole")
plt.xlabel("x (cm)")
plt.ylabel("y (cm)")

# Mark charge locations
plt.scatter([pos_q[0], neg_q[0]], [pos_q[1], neg_q[1]], c=["red", "blue"], marker="o", s=100, label="Charges")
plt.legend(["Negative Charge", "Positive Charge"])
plt.show()

###################################################################
# Problem 3 (a & b)
# Solving Matrices
###################################################################
#Exercises 6.1 in your book shows a network of resistors and suggests a method 
#to solve for V at each point. Write out the full system of equations and then 
#implement the code to solve them using Gaussian elimination.
#
