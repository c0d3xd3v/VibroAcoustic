from ngsolve import *
from netgen.stl import *
from netgen.csg import *
from netgen.meshing import *
import netgen.meshing as nm


from netgen.read_gmsh import ReadGmsh
from ngsolve import Mesh as NGSMesh

import numpy as np

def computeMeshCenter(mesh):
    centerx = 0
    centery = 0
    centerz = 0
    count   = 0

    for e in mesh.Elements2D():
        for v in e.vertices:
            centerx = centerx + mesh[v][0]
            centery = centery + mesh[v][1]
            centerz = centerz + mesh[v][2]
            count = count + 1

    centerx = centerx / count
    centery = centery / count
    centerz = centerz / count

    return [centerx, centery, centerz]

def meshBoundingRadius(mesh, center):
    R = 0
    for e in mesh.Elements2D():
        for v in e.vertices:
                x = -center[0] + mesh[v][0]
                y = -center[1] + mesh[v][1]
                z = -center[2] + mesh[v][2]
                tmp = sqrt(x*x + y*y + z*z)
                if tmp > R:
                    R = tmp
    return R

def addDomain(mesh, domainMesh, fd):
    pmap = {}
    for e in domainMesh.Elements2D():
        for v in e.vertices:
            if v not in pmap:
                pmap[v] = mesh.Add(domainMesh[v])
    # copy surface elements from first mesh to new mesh
    # we have to map point-numbers:
    for e in domainMesh.Elements2D():
        mesh.Add(Element2D(fd, [pmap[v] for v in e.vertices]))
    return mesh

def addDomainSave(mesh, domainMesh, fd, fd_fixed):
    pmap = {}
    for e in domainMesh.Elements2D():
        for v in e.vertices:
            if v not in pmap:
                pmap[v] = mesh.Add(domainMesh[v])
    # copy surface elements from first mesh to new mesh
    # we have to map point-numbers:
    for e in domainMesh.Elements2D():
        #fd = domainMesh.FaceDescriptors()[e.index]
        if(e.index == 1):
            mesh.Add(Element2D(fd_fixed, [pmap[v] for v in e.vertices]))
        else:
            mesh.Add(Element2D(fd, [pmap[v] for v in e.vertices]))
    return mesh

def addSurfaceFromLists(fds, mesh, pmap, tris):
    for i, tri in enumerate(tris):
        vindices = [pmap[v] for v in tri]
        T = Element2D(fds, vindices)
        mesh.Add(T)
    return mesh

def compute_tet_volume(tet):
        v0, v1, v2, v3 = tet
        vol = np.dot(np.cross(v1 - v0, v3 - v0), v2 - v0)
        #print(f'vol: {vol}')
        return vol

def addVolumeFromLists(index, mesh, pmap, tets):
    for i, tet in enumerate(tets):
        vindices = [pmap[v] for v in tet]
        vindices[2], vindices[3] = vindices[3], vindices[2]

        tet = [np.array([mesh[v][0], mesh[v][1], mesh[v][2]]) for v in vindices]
        vol = compute_tet_volume(tet)
        #if vol < 0:
            #vindices[2], vindices[3] = vindices[3], vindices[2]
            #tet = [np.array([mesh[v][0], mesh[v][1], mesh[v][2]]) for v in vindices]
            #v0, v1, v2, v3 = tet
            #vol = np.dot(np.cross(v1 - v0, v2 - v0), v3 - v0)
            #print(f'vol : {vol}')

        if vol != 0.0:
            T = Element3D(index, vindices)
            mesh.Add(T)
    return mesh

def saveNgSolveMesh(nods, tris, tets, file_name):
    # Aufbau der Netgen-Mesh
    mesh = nm.Mesh()
    pmap = {}
    for i, p in enumerate(nods):
        mp = MeshPoint(Point3d(p[0], p[1], p[2]))
        pmap[i] = mesh.Add(mp)

    fds = mesh.Add(FaceDescriptor(bc=1, domin=0, surfnr=0))
    mesh = addSurfaceFromLists(fds, mesh, pmap, tris)
    mesh = addVolumeFromLists(0, mesh, pmap, tets)
    mesh.Update()
    mesh.Save(file_name)