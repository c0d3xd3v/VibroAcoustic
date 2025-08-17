bl_info = {
    "name": "Colormap Loader",
    "blender": (3, 0, 0),
    "category": "Node",
}

import bpy
import xml.etree.ElementTree as ET


# Funktion zum Laden der Colormap aus XML
def load_colormap_from_xml(file_path):
    colormap = []
    tree = ET.parse(file_path)
    root = tree.getroot()

    for point in root.findall('Point'):
        position = float(point.get('x'))
        r = float(point.get('r'))
        g = float(point.get('g'))
        b = float(point.get('b'))
        colormap.append((position, (r, g, b)))

    return colormap


# Operator zum Erstellen der Colormap im Shader
class NODE_OT_AddColormap(bpy.types.Operator):
    bl_idname = "node.add_colormap"
    bl_label = "Add Colormap from XML"
    bl_options = {'REGISTER', 'UNDO'}

    # Verwende FilePath, um den Dateipfad auszuwählen
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    def execute(self, context):
        # Die XML-Datei laden
        colormap = load_colormap_from_xml(self.filepath)

        # Den aktiven Materialnodebaum holen
        material = bpy.context.object.active_material
        if not material:
            self.report({'ERROR'}, "No active material found!")
            return {'CANCELLED'}

        node_tree = material.node_tree
        nodes = node_tree.nodes

        # ColorRamp-Node hinzufügen
        color_ramp_node = nodes.new(type='ShaderNodeValToRGB')
        color_ramp_node.location = (300, 300)

        # Clear existing elements in ColorRamp
        color_ramp = color_ramp_node.color_ramp

        # Remove all existing elements in reverse order to avoid index issues
        n = len(color_ramp.elements) - 1
        for i in range(n):
            self.report({'INFO'}, str(len(color_ramp.elements)))
            color_ramp.elements.remove(color_ramp.elements[-1])  # Entferne das letzte Element

        # ColorRamp-Positionen basierend auf der Colormap einstellen
        for position, color in colormap:
            element = color_ramp.elements.new(position)
            element.color = (*color, 1.0)  # (r, g, b, alpha)

        color_ramp.elements.remove(color_ramp.elements[0])

        self.report({'INFO'}, "Colormap loaded successfully!")
        return {'FINISHED'}

    # Diese Methode öffnet den Dateiauswahldialog
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


# Panel zum Hinzufügen des Operators
class NODE_PT_ColormapLoader(bpy.types.Panel):
    bl_idname = "NODE_PT_colormap_loader"
    bl_label = "Colormap Loader"
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Tools"

    def draw(self, context):
        layout = self.layout
        layout.operator(NODE_OT_AddColormap.bl_idname)


# Registrierung der Klassen
def register():
    bpy.utils.register_class(NODE_OT_AddColormap)
    bpy.utils.register_class(NODE_PT_ColormapLoader)


def unregister():
    bpy.utils.unregister_class(NODE_OT_AddColormap)
    bpy.utils.unregister_class(NODE_PT_ColormapLoader)


if __name__ == "__main__":
    register()
