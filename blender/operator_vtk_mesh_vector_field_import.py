bl_info = {
    "name": "vtk mesh with vector field",
    "author": "Kai Henning",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "File > Import",
    "description": "import vtk mesh with vector field",
    "category": "Import-Export",
}

import bpy
import vtk
import numpy as np


class IMPORT_OT_custom_file(bpy.types.Operator):
    """VTK File Import Operator"""
    bl_idname = "import_scene.custom_file"
    bl_label = "Import VTK File"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    def execute(self, context):
        # Implement the logic for your custom file import here
        self.report({'INFO'}, "File Imported: " + self.filepath)
        polydata = self.load_vtk_unstructured_grid(self.filepath)
        self.create_mesh_with_vector_fields_as_attribute(polydata, "vtkmeshsurface")
        
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def load_vtk_unstructured_grid(self, vtk_file_path):
        # Read the VTK file (assuming it's an unstructured grid)
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(vtk_file_path)
        reader.Update()

        # Get the unstructured grid output
        unstructured_grid = reader.GetOutput()

        # Use vtkDataSetSurfaceFilter to extract the surface geometry (triangular mesh)
        surface_filter = vtk.vtkDataSetSurfaceFilter()
        surface_filter.SetInputData(unstructured_grid)
        surface_filter.Update()

        polydata = surface_filter.GetOutput()

        # List available arrays
        point_data = polydata.GetPointData()
        num_arrays = point_data.GetNumberOfArrays()

        print("Number of Arrays:", num_arrays)
        for i in range(num_arrays):
            array_name = point_data.GetArrayName(i)
            array_type = point_data.GetArray(i).GetDataTypeAsString()
            print(f"Array {i}: Name = {array_name}, Type = {array_type}")

        return polydata


    def get_vector_data_from_array(self, polydata, array_index):
        # Get the data from the specified array index
        point_data = polydata.GetPointData()
        array_data = point_data.GetArray(array_index)
        
        if array_data:
            num_tuples = array_data.GetNumberOfTuples()
            vector_data = np.array([array_data.GetTuple(i) for i in range(num_tuples)])
            return vector_data
        return None
    
    def add_vector_field_attribute(self, mesh, vector_values, field_name):

        max_norm = np.max(np.linalg.norm(vector_values, axis=1))

        # create vector valued point attribute
        if field_name not in mesh.attributes:    
            mesh.attributes.new(name=field_name, type='FLOAT_VECTOR', domain='POINT')

        attr = mesh.attributes[field_name].data

        # add normalized Vector field data
        num_vertices = len(mesh.vertices)
        if vector_values.shape[0] == num_vertices:
            for i in range(num_vertices):
                attr[i].vector = vector_values[i]/max_norm
            print(f"Vector field data added as attribute 'VectorField'.")
        else:
            print("Error: The number of vectors does not match the number of vertices.")  


    def create_mesh_with_vector_fields_as_attribute(self, polydata, mesh_name):
        # Get vertices and triangles from polydata 
        vertices = []
        triangles = []
        vector_fields = []

        # Extract vertices
        for i in range(polydata.GetNumberOfPoints()):
            point = polydata.GetPoint(i)
            vertices.append((point[0], point[1], point[2]))

        # Extract faces (triangles)
        for i in range(polydata.GetNumberOfCells()):
            cell = polydata.GetCell(i)
            if cell.GetNumberOfPoints() == 3:  # We only care about triangles
                triangles.append((
                    cell.GetPointId(0),
                    cell.GetPointId(1),
                    cell.GetPointId(2)
                ))
                
        point_data = polydata.GetPointData()
        num_arrays = point_data.GetNumberOfArrays()
        self.report({'INFO'}, "num_arrays: " + str(num_arrays))

        for i in range(num_arrays):
            vector_fields.append(self.get_vector_data_from_array(polydata, i))

        # Create a new mesh and object
        mesh = bpy.data.meshes.new(mesh_name)
        obj = bpy.data.objects.new(mesh_name, mesh)
        
        # Link the object to the active collection
        bpy.context.collection.objects.link(obj)

        # Create the mesh from the provided data
        mesh.from_pydata(vertices, [], triangles)
        mesh.update()
        
        # add vector fields as attribute
        for i, vector_field in enumerate(vector_fields):
            field_name = "mode"+str(i)
            self.add_vector_field_attribute(mesh, vector_field, field_name)


######################################################################################################
def menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_custom_file.bl_idname, text="vtk mesh with vector field (.vtk)")

def register():
    bpy.utils.register_class(IMPORT_OT_custom_file)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)

def unregister():
    bpy.utils.unregister_class(IMPORT_OT_custom_file)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)

if __name__ == "__main__":
    register()
