#!/usr/bin/env rotation matrix crispey
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sqlite3
import os

#data management, simple sql:
def create_db():
    if not os.path.exists('rotation_results.db'):
        conn = sqlite3.connect('rotation_results.db')  # Create or open the database file
        cursor = conn.cursor()
        
        cursor.execute('DROP TABLE IF EXISTS rotation_data')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS rotation_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            SROT REAL,
            phi REAL,
            Ev REAL, 
            k REAL
        )
        ''')
        
        cursor.execute('PRAGMA table_info(rotation_data);')
        columns = cursor.fetchall()
        for column in columns:
            print(column)

        conn.commit()
        conn.close()
        
        print("Database and table created successfully.")
    else:
        print("Database already exists. Using the existing database.")

# extracting Ev and k as strings in format of crispy input
def insert_data(SROT, phi, Ev, k):
    conn = sqlite3.connect('rotation_results.db')  # Connect to the database
    cursor = conn.cursor()
    Ev_str = ', '.join(map(str, Ev))
    k_str = ', '.join(map(str, k))

    cursor.execute('''
    INSERT INTO rotation_data (SROT, phi, Ev, k)
    VALUES (?, ?, ?, ?)
    ''', (SROT, phi, Ev_str, k_str))

    conn.commit()
    conn.close()

# rotation matrices declared
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

#rotate function 
def rotate_vector(vector, SROT, phi):
    R_srot = rotation_matrix_y(SROT)
    R_phi = rotation_matrix_z(phi)
    return R_phi @ R_srot @ vector

# projection function
def project_onto(v, direction):
    norm_direction = np.linalg.norm(direction)
    if norm_direction == 0:
        return np.zeros_like(v)
    return np.dot(v, direction) / norm_direction**2 * direction

# crystal field for input (relevant rarely)
def calculate_EF_components(EF_initial, SROT, phi, magnitude):
    EF_rotated = rotate_vector(EF_initial, SROT, phi)
    EF_normalized = EF_rotated / np.linalg.norm(EF_rotated) 
    EF_scaled = EF_normalized * magnitude             
    return EF_scaled[0], EF_scaled[1], EF_scaled[2]

# E field calc
def calculate_Ev_Eh(some_k, EF):
    z_axis = np.array([-1, 1, 0])  # im projecting all onto z plane, because it works best. Can project onto 111 but i feel like it is the simplest. 
    Ev = np.cross(some_k, z_axis) 
    Ev = Ev / np.linalg.norm(Ev)

    Eh = np.cross(some_k, Ev)
    Eh = Eh / np.linalg.norm(Eh)

    return Ev, Eh
def EF_final(EFx, EFy, EFz):

    return np.array([EFx, EFy, EFz])

# k_ initial in our geometry is always 00-1, EF accordingly to the experimental setup
k_initial = np.array([0, 0, -1]) 
# declarations in case of exception
EF_initial = np.array([4, 2, 0]) 
SROT = 90
phi_values = [0, 6, 9]
EF_magnitude = 1.40 

print(f"Initial EF-vector components (no normalization):")
print(f"EF_x: {EF_initial[0]:.3f}, EF_y: {EF_initial[1]:.3f}, EF_z: {EF_initial[2]:.3f}\n")

# 
fig, axes = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(18, 6))

SROT = float(input("Enter SROT angle (in degrees, e.g.: -45): "))
phi_values = list(map(float, input("Enter phi values (comma-separated, up to three, e.g.: 30, 60, 90): ").split(',')))
EF_magnitude = float(input("Enter desired EF magnitude (e.g.: for NiO, meV: 1.42 : "))
EF_initial = np.array(list(map(float, input("Enter M initial vector (comma-separated, e.g.:, 4,1,3): ").split(','))))
create_db()
# initially for multiple phis, as usually we do azimunthal plots in fixed polar angle 

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
    print(f"since srot is {SROT} and z is existing: {EF_z} rotate via changing k and Ev for initially set and constant CF = [1,1,0] which is a projection plane thank you")
    insert_data(SROT, phi, Ev, k_perpendicular_rotated) 

    # subplots
    ax = axes[i]
    
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', linewidth=2)  # x-axis (red)
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', linewidth=2)  # y-axis (green)
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', linewidth=2)  # z-axis (blue)

    # initial crystal field input
    ax.quiver(0, 0, 0, EF_initial[0], EF_initial[1], EF_initial[2], color='m', linewidth=2, label='EF initial')

    # and rotated cf 
    ax.quiver(0, 0, 0, EF_x, EF_y, EF_z, color='m', linestyle='dashed', linewidth=2, label=f'EF final scaled ({EF_x:.3f}, {EF_y:.3f}, {EF_z:.3f})')

    ax.quiver(0, 0, 0, k_perpendicular_rotated[0], k_perpendicular_rotated[1], k_perpendicular_rotated[2], color='y', linestyle='dotted', linewidth=2, label=f'Perpendicular component rotated ({k_perpendicular_rotated[0]:.3f}, {k_perpendicular_rotated[1]:.3f}, {k_perpendicular_rotated[2]:.3f})')

    k_final_proj_on_EF = project_onto(k_final_updated, EF_rotated)

    #vertical e (to check if consistent with crispy)
    ax.quiver(0, 0, 0, Ev[0], Ev[1], Ev[2], color='black', linewidth=2, label=f'Ev_calculated ({Ev[0]:.3f}, {Ev[1]:.3f}, {Ev[2]:.3f})')
    
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
    if EF_z == 0 and SROT == 0:
     print(f"since srot is {SROT} and z is {EF_z} rotate via changing CF for initially set k = [0,0,-1], Eh = [1,0,0] and Ev = [0,1,0], or however Ev and Eh should be oriented in your specific case. Thanks")
    print()


#initial dir calc
def calculate_xy_angle(v1, v2):
    v1_xy = v1[:2] 
    v2_xy = v2[:2] 
    
    return calculate_angle(v1_xy, v2_xy)

def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    angle_radians = np.arccos(cos_theta)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

dir_100 = np.array([1, 0, 0])  #  (x-axis)
dir_010 = np.array([0, 1, 0])  # (y-axis)
dir_110 = np.array([1, -1, 0])  # (xy plane)

angle_100_xy = calculate_xy_angle(Ev, dir_100)
angle_010_xy = calculate_xy_angle(Ev, dir_010)
angle_110_xy = calculate_xy_angle(Ev, dir_110)

print(f"XY Angle between k and [100] direction: {angle_100_xy:.2f} degrees")
print(f"XY Angle between k and [010] direction: {angle_010_xy:.2f} degrees")
print(f"XY Angle between k and [110] direction: {angle_110_xy:.2f} degrees")

fig, ax = plt.subplots(figsize=(6, 6))

ax.quiver(0, 0, Ev[0], Ev[1], angles='xy', scale_units='xy', scale=1, color='b', label='Ev')
ax.quiver(0, 0, dir_100[0], dir_100[1], angles='xy', scale_units='xy', scale=1, color='r', label='[100]')
ax.quiver(0, 0, dir_010[0], dir_010[1], angles='xy', scale_units='xy', scale=1, color='g', label='[010]')
ax.quiver(0, 0, dir_110[0], dir_110[1], angles='xy', scale_units='xy', scale=1, color='g', label='[-110]')

ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_aspect('equal')
ax.legend()
plt.title("XY-plane Vector Plot")
plt.grid(True)
plt.show()


