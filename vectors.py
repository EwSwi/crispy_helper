#!/usr/bin/env rotation matrix crispey

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Rotation matrix for rotating around x, y, and z axes
def rotation_matrix_x(angle):
    return np.array([
        [1, 0, 0],
        [0, np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
        [0, np.sin(np.radians(angle)), np.cos(np.radians(angle))]
    ])

def rotation_matrix_y(angle):
    return np.array([
        [np.cos(np.radians(angle)), 0, np.sin(np.radians(angle))],
        [0, 1, 0],
        [-np.sin(np.radians(angle)), 0, np.cos(np.radians(angle))]
    ])

def rotation_matrix_z(angle):
    return np.array([
        [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
        [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0],
        [0, 0, 1]
    ])

# Rotation function (rotation matrices used)
def rotate_vector(vector, SROT, phi):
    R_srot = rotation_matrix_y(SROT)
    R_phi = rotation_matrix_z(phi)
    return R_phi @ R_srot @ vector

# Projection to project rotated k onto rotated EF
def project_onto(v, direction):
    norm_direction = np.linalg.norm(direction)
    if norm_direction == 0:
        return np.zeros_like(v)
    return np.dot(v, direction) / norm_direction**2 * direction

# Function to calculate EF components after rotation and scaling by a given magnitude
def calculate_EF_components(EF_initial, SROT, phi, magnitude):
    EF_rotated = rotate_vector(EF_initial, SROT, phi)
    EF_normalized = EF_rotated / np.linalg.norm(EF_rotated)  # Normalize the EF vector
    EF_scaled = EF_normalized * magnitude                    # Scale it by the desired magnitude
    return EF_scaled[0], EF_scaled[1], EF_scaled[2]

# Calculate Ev and Eh based on the rotated k vector
def calculate_Ev_Eh(some_k, EF):
    z_axis = np.array([-1, 1, 0])  # Z-axis for vertical direction
    Ev = np.cross(some_k, z_axis)  # Vertical component perpendicular to k and z-axis
    Ev = Ev / np.linalg.norm(Ev)  # Normalize

    Eh = np.cross(some_k, Ev)  # Horizontal component perpendicular to both k and Ev
    Eh = Eh / np.linalg.norm(Eh)  # Normalize

    return Ev, Eh
def EF_final(EFx, EFy, EFz):

    return np.array([EFx, EFy, EFz])

# k_ initial in our geometry is always 00-1, EF accordingly to the experimental setup
k_initial = np.array([0, 0, -1]) 
# EF is the value one is supposed to change  
EF_initial = np.array([4, 1, 3]) 

# These two values are inputs
SROT = 90
phi_values = [0, 45, 90]
EF_magnitude = 1.40 # Desired EF magnitude

print(f"Initial EF-vector components (no normalization):")
print(f"EF_x: {EF_initial[0]:.3f}, EF_y: {EF_initial[1]:.3f}, EF_z: {EF_initial[2]:.3f}\n")

# Plotting
fig, axes = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(18, 6))

SROT = float(input("Enter SROT angle (in degrees): "))
phi_values = list(map(float, input("Enter phi values (comma-separated): ").split(',')))
EF_magnitude = float(input("Enter desired EF magnitude: "))
EF_initial = np.array(list(map(float, input("Enter M initial vector (comma-separated, e.g., 4,1,3): ").split(','))))
# Plotting multiple phis
for i, phi in enumerate(phi_values):
    k_rotated = rotate_vector(k_initial, SROT, phi)
    EF_rotated = rotate_vector(EF_initial, SROT, phi)

    k_proj_on_EF = project_onto(k_rotated, EF_rotated)
    k_perpendicular_rotated = k_rotated - k_proj_on_EF
    k_final_updated = k_proj_on_EF + k_perpendicular_rotated

    h_horizontal_rotated = rotate_vector(np.array([1, 0, 0]), SROT, phi)
    v_vertical_rotated = rotate_vector(np.array([0, 1, 0]), SROT, phi)
    
    # Get EF components using the new function with magnitude
    EF_x, EF_y, EF_z = calculate_EF_components(EF_initial, SROT, phi, EF_magnitude)
    EF_final_vector = EF_final(EF_x, EF_y, EF_z)
    Ev, Eh = calculate_Ev_Eh(k_perpendicular_rotated, EF_final_vector)

    # Plotting for subplots
    ax = axes[i]
    
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', linewidth=2)  # x-axis (red)
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', linewidth=2)  # y-axis (green)
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', linewidth=2)  # z-axis (blue)

    # Initial plot, if one does not need it just hash it
    #ax.quiver(0, 0, 0, k_initial[0], k_initial[1], k_initial[2], color='k', linewidth=2, label='k initial')
    ax.quiver(0, 0, 0, EF_initial[0], EF_initial[1], EF_initial[2], color='m', linewidth=2, label='EF initial')

    # Final vectors
    #ax.quiver(0, 0, 0, k_final_updated[0], k_final_updated[1], k_final_updated[2], color='k', linestyle='dashed', linewidth=2, label=f'k final ({k_final_updated[0]:.3f}, {k_final_updated[1]:.3f}, {k_final_updated[2]:.3f})')
    ax.quiver(0, 0, 0, EF_x, EF_y, EF_z, color='m', linestyle='dashed', linewidth=2, label=f'EF final scaled ({EF_x:.3f}, {EF_y:.3f}, {EF_z:.3f})')

    # Visualization vector
   # ax.quiver(0, 0, 0, k_proj_on_EF[0], k_proj_on_EF[1], k_proj_on_EF[2], color='c', linestyle='dotted', linewidth=2, label='k projected onto EF')

    ax.quiver(0, 0, 0, k_perpendicular_rotated[0], k_perpendicular_rotated[1], k_perpendicular_rotated[2], color='y', linestyle='dotted', linewidth=2, label=f'Perpendicular component rotated ({k_perpendicular_rotated[0]:.3f}, {k_perpendicular_rotated[1]:.3f}, {k_perpendicular_rotated[2]:.3f})')

    k_final_proj_on_EF = project_onto(k_final_updated, EF_rotated)
    #ax.quiver(0, 0, 0, k_final_proj_on_EF[0], k_final_proj_on_EF[1], k_final_proj_on_EF[2], color='purple', linestyle='dashdot', linewidth=2, label=f'k_final projected onto EF_rotated ({k_final_proj_on_EF[0]:.3f}, {k_final_proj_on_EF[1]:.3f}, {k_final_proj_on_EF[2]:.3f})')

    # Add "crispy" vector
   # ax.quiver(0, 0, 0, crispy[0], crispy[1], crispy[2], color='orange', linewidth=2, label=f'crispy_Ev ({crispy[0]:.3f}, {crispy[1]:.3f}, {crispy[2]:.3f})')
    #ax.quiver(0, 0, 0, Eh[0], Eh[1], Eh[2], color='red', linewidth=2, label=f'Eh_calculated ({Eh[0]:.3f}, {Eh[1]:.3f}, {Eh[2]:.3f})')
    ax.quiver(0, 0, 0, Ev[0], Ev[1], Ev[2], color='black', linewidth=2, label=f'Ev_calculated ({Ev[0]:.3f}, {Ev[1]:.3f}, {Ev[2]:.3f})')

    # Plot aesthetics
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title(f'SROT {SROT} deg, phi = {phi} deg')
    ax.set_box_aspect([1, 1, 1])  
    ax.legend()
    ax.grid(True)

    # Prints
    print(f"phi: {phi}")
    print(f"EF components after rotation and scaling: EF_x = {EF_x:.3f}, EF_y = {EF_y:.3f}, EF_z = {EF_z:.3f}")
    print(f"Projection of rotated k onto rotated EF: {k_proj_on_EF}")
    print(f"Perpendicular Component of rotated k: {k_perpendicular_rotated}")
    print(f"Projection of k_final_updated onto EF_rotated: {k_final_proj_on_EF}")
    print(f"E_vertical = {Ev}")
    print(f"E_horizontal = {Eh}")
    print()

# Show the plot
plt.tight_layout()
plt.show()
