"""Render a donut with Blender, headless.

    blender -b --factory-startup --python blender/donut.py -- --out donut.png

Builds the scene from scratch (dough, icing, sprinkles, plate, camera, lights)
and renders one PNG with Cycles on the CPU. Options after "--":

  --out PATH     output PNG (default: donut.png)
  --samples N    Cycles samples (default: 64)
  --size WxH     resolution (default: 1280x720)
"""

import argparse
import math
import os
import random
import sys

import bpy
from mathutils import Matrix, Vector

R, r = 1.0, 0.42  # dough torus: ring radius, tube radius
ICING = r + 0.025  # icing shell sits just outside the dough


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="donut.py")
    p.add_argument("--out", default="donut.png")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--size", default="1280x720")
    return p.parse_args(argv)


def material(name, color, roughness):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def smooth(obj, mat):
    obj.data.materials.append(mat)
    for poly in obj.data.polygons:
        poly.use_smooth = True
    return obj


def add_object(name, data, location=(0, 0, 0)):
    obj = bpy.data.objects.new(name, data)
    obj.location = location
    bpy.context.collection.objects.link(obj)
    return obj


def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def icing_edge(theta):
    """Height above which the icing covers the dough; the waves make it drip."""
    return 0.05 + 0.06 * math.sin(5 * theta) + 0.015 * math.sin(13 * theta + 1.3)


def icing_mesh(rings=256, steps=64):
    """The part of the icing torus above icing_edge, with rows that follow the edge curve."""
    verts, faces = [], []
    for i in range(rings):
        theta = 2 * math.pi * i / rings
        lo = math.asin(min(icing_edge(theta) / ICING, 0.99))
        for j in range(steps + 1):
            phi = lo + (math.pi - 2 * lo) * j / steps  # outer edge, over the top, to the inner edge
            ring = R + ICING * math.cos(phi)
            verts.append((ring * math.cos(theta), ring * math.sin(theta), ICING * math.sin(phi)))
    for i in range(rings):
        a, b = i * (steps + 1), (i + 1) % rings * (steps + 1)
        faces += [(a + j, b + j, b + j + 1, a + j + 1) for j in range(steps)]  # wound so normals face out
    mesh = bpy.data.meshes.new("Icing")
    mesh.from_pydata(verts, [], faces)
    return mesh


def build_donut():
    bpy.ops.mesh.primitive_torus_add(major_radius=R, minor_radius=r, major_segments=96, minor_segments=48)
    dough = smooth(bpy.context.active_object, material("Dough", (0.62, 0.33, 0.13), 0.65))
    dough.modifiers.new("Subdivision", "SUBSURF").levels = 1

    icing = smooth(add_object("Icing", icing_mesh()), material("Icing", (0.85, 0.22, 0.42), 0.25))
    solidify = icing.modifiers.new("Solidify", "SOLIDIFY")
    solidify.thickness = 0.03
    solidify.offset = -1
    icing.modifiers.new("Subdivision", "SUBSURF").levels = 1


def build_sprinkles(count=140, seed=7):
    colors = [(0.95, 0.8, 0.15), (0.15, 0.45, 0.95), (0.25, 0.75, 0.3), (0.95, 0.95, 0.92), (0.95, 0.4, 0.08)]
    meshes = []
    for i, color in enumerate(colors):
        bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=0.016, depth=0.11)
        obj = smooth(bpy.context.active_object, material(f"Sprinkle {i}", color, 0.4))
        meshes.append(obj.data)
        bpy.data.objects.remove(obj, do_unlink=True)

    rng = random.Random(seed)
    lift = ICING + 0.012
    placed = 0
    while placed < count:
        theta = rng.uniform(0, 2 * math.pi)
        phi = rng.uniform(0.15, math.pi - 0.15)  # 0 = outer equator, pi/2 = top, pi = inner equator
        if ICING * math.sin(phi) < icing_edge(theta) + 0.08:
            continue  # keep sprinkles on the icing, clear of its edge
        normal = Vector((math.cos(phi) * math.cos(theta), math.cos(phi) * math.sin(theta), math.sin(phi)))
        center = Vector((R * math.cos(theta), R * math.sin(theta), 0)) + lift * normal
        along_ring = Vector((-math.sin(theta), math.cos(theta), 0))
        a = rng.uniform(0, math.pi)
        axis = (math.cos(a) * along_ring + math.sin(a) * normal.cross(along_ring)).normalized()
        rotation = Matrix((normal.cross(axis), normal, axis)).transposed().to_4x4()
        add_object(f"Sprinkle {placed}", rng.choice(meshes)).matrix_world = Matrix.Translation(center) @ rotation
        placed += 1


def build_set(scene):
    bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=1.9, depth=0.06, location=(0, 0, -r - 0.03))
    bpy.context.active_object.data.materials.append(material("Plate", (0.9, 0.9, 0.88), 0.2))
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, -r - 0.06))
    bpy.context.active_object.data.materials.append(material("Table", (0.3, 0.4, 0.48), 0.8))

    camera = add_object("Camera", bpy.data.cameras.new("Camera"), (0, -6.3, 4.8))
    camera.data.lens = 50
    aim(camera, (0, 0, -0.3))
    scene.camera = camera

    for name, energy, size, location in [
        ("Key", 700, 4, (3.5, -3, 5)),
        ("Fill", 150, 6, (-5, -2, 3)),
        ("Rim", 300, 3, (0, 5, 3)),
    ]:
        light = bpy.data.lights.new(name, "AREA")
        light.energy = energy
        light.size = size
        aim(add_object(name, light, location), (0, 0, 0))

    background = scene.world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.05, 0.06, 0.08, 1)
    background.inputs["Strength"].default_value = 1.0


def render(scene, args):
    import _cycles

    width, height = (int(v) for v in args.size.lower().split("x"))
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    # Some Linux distro builds of Blender ship without OpenImageDenoise.
    scene.cycles.use_denoising = bool(getattr(_cycles, "with_openimagedenoise", False))
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = os.path.abspath(args.out)
    bpy.ops.render.render(write_still=True)


def main():
    args = parse_args()
    scene = bpy.context.scene
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    build_donut()
    build_sprinkles()
    build_set(scene)
    render(scene, args)


main()
