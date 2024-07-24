# 3D-printing-using-ML-AI-
Custom 3D printing of Coffee mug using ML/AI
pip install numpy-stl numpy
import numpy as np
from stl import mesh

# Parameters
height = 100    # Height of the mug
outer_radius = 40   # Outer radius of the mug
thickness = 3   # Wall thickness of the mug
handle_radius = 10  # Radius of the handle's tube
handle_thickness = 5 # Thickness of the handle's tube
handle_length = 30   # Length of the handle

# Function to create a cylinder
def create_cylinder(radius, height, num_segments):
    theta = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    
    vertices = []
    faces = []
    
    for i in range(num_segments):
        # Bottom face
        vertices.append([x[i], y[i], 0])
        vertices.append([x[(i + 1) % num_segments], y[(i + 1) % num_segments], 0])
        vertices.append([0, 0, 0])
        
        # Top face
        vertices.append([x[i], y[i], height])
        vertices.append([x[(i + 1) % num_segments], y[(i + 1) % num_segments], height])
        vertices.append([0, 0, height])
        
        # Side faces
        vertices.append([x[i], y[i], 0])
        vertices.append([x[(i + 1) % num_segments], y[(i + 1) % num_segments], 0])
        vertices.append([x[(i + 1) % num_segments], y[(i + 1) % num_segments], height])
        
        vertices.append([x[i], y[i], 0])
        vertices.append([x[(i + 1) % num_segments], y[(i + 1) % num_segments], height])
        vertices.append([x[i], y[i], height])
    
    faces = np.arange(len(vertices)).reshape(-1, 3)
    return np.array(vertices), faces

# Create the outer and inner cylinders for the mug body
outer_vertices, outer_faces = create_cylinder(outer_radius, height, 64)
inner_vertices, inner_faces = create_cylinder(outer_radius - thickness, height, 64)
inner_vertices[:, 2] += thickness

# Combine the vertices and faces to form the mug body
vertices = np.vstack([outer_vertices, inner_vertices])
faces = np.vstack([
    outer_faces, 
    inner_faces + len(outer_vertices)
])

# Function to create a handle (a torus-like shape)
def create_handle():
    theta = np.linspace(0, 2 * np.pi, 64)
    phi = np.linspace(0, 2 * np.pi, 64)
    theta, phi = np.meshgrid(theta, phi)
    
    x = (handle_radius + handle_thickness * np.cos(phi)) * np.cos(theta)
    y = (handle_radius + handle_thickness * np.cos(phi)) * np.sin(theta)
    z = handle_thickness * np.sin(phi)
    
    vertices_handle = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    num_theta = len(theta)
    num_phi = len(phi)
    
    faces_handle = []
    
    for i in range(num_theta - 1):
        for j in range(num_phi - 1):
            faces_handle.append([i * num_phi + j, (i + 1) * num_phi + j, i * num_phi + j + 1])
            faces_handle.append([(i + 1) * num_phi + j, (i + 1) * num_phi + j + 1, i * num_phi + j + 1])
    
    return np.array(vertices_handle), np.array(faces_handle)

handle_vertices, handle_faces = create_handle()
handle_vertices += [outer_radius + handle_thickness / 2, 0, height / 2]

# Combine the mug and handle
combined_vertices = np.vstack([vertices, handle_vertices])
combined_faces = np.vstack([faces, handle_faces + len(vertices)])

# Create the mesh
mug_mesh = mesh.Mesh(np.zeros(combined_faces.shape[0], dtype=mesh.Mesh.dtype))
for i, face in enumerate(combined_faces):
    for j in range(3):
        mug_mesh.vectors[i][j] = combined_vertices[face[j], :]

# Save the STL file
mug_mesh.save('coffee_mug.stl')
print("Coffee mug model has been saved as 'coffee_mug.stl'")
