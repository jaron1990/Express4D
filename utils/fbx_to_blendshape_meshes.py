

import bpy
import os

# Path to your FBX file
fbx_file_path = "/home/dcor/yaronaloni/Express4D/dataset/metaHumanHead_52shapekeys_01.fbx"

# Directory to save the OBJ files
output_directory = "/home/dcor/yaronaloni/Express4D/obj_exports/"
os.makedirs(output_directory, exist_ok=True)

# Import the FBX file
bpy.ops.import_scene.fbx(filepath=fbx_file_path)

# Remove the default cube if present
for scene_obj in bpy.context.scene.objects:
    if scene_obj.type == 'MESH' and scene_obj.name.lower() == 'cube':
        bpy.data.objects.remove(scene_obj, do_unlink=True)

# Helper function to write OBJ files
def write_obj_file(vertices, faces, file_path):
    with open(file_path, 'w') as f:
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        # Write faces
        for face in faces:
            f.write(f"f {' '.join([str(idx + 1) for idx in face])}\n")

    print(f"Exported mesh to {file_path}")

# Process all objects with shape keys
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH' and obj.data.shape_keys:
        shape_keys = obj.data.shape_keys.key_blocks
        shape_key_names = list(shape_keys.keys())

        print(f"Processing object: {obj.name}")
        
        # Export the base mesh ('Basis')
        basis_vertices = [[v.co.x, v.co.y, v.co.z] for v in shape_keys['Basis'].data]
        faces = [list(poly.vertices) for poly in obj.data.polygons]
        base_file_path = os.path.join(output_directory, f"{obj.name}_Basis.obj")
        write_obj_file(basis_vertices, faces, base_file_path)

        # Iterate over all blendshapes
        for blendshape_name in shape_key_names[1:]:  # Skip 'Basis'
            blendshape_vertices = [[v.co.x, v.co.y, v.co.z] for v in shape_keys[blendshape_name].data]

            # Export the blendshape vertices to an OBJ file
            blendshape_file_path = os.path.join(output_directory, f"{obj.name}_{blendshape_name}_1.0.obj")
            write_obj_file(blendshape_vertices, faces, blendshape_file_path)
