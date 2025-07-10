from mpi4py import MPI
from dolfinx import mesh, fem, io, plot, nls, log, geometry, la
import basix.ufl
import dolfinx.fem.petsc
import ufl
import numpy as np
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc
import time
import os
import gmsh


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# Parameters for the geometry

L = 50

H = 2.5
B = 0.5

ac = 10

eps = 0.16
h = 0.05

make_mesh = True

if make_mesh:
    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
        gmsh.model.add("Pureshear")


        # Create outer box
        block = gmsh.model.occ.addBox(0, 0, 0, L, H, -B)

        # Synchronize to reflect the changes in the model
        gmsh.model.occ.synchronize()

        # Add physical group for the volume (the tube itself)
        gmsh.model.addPhysicalGroup(3, [block], 1)
        gmsh.model.setPhysicalName(3, 1, "block")
        


        # Define mesh size fields
        field_id = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(field_id, "VIn", h)
        gmsh.model.mesh.field.setNumber(field_id, "VOut", 8 * h)
        gmsh.model.mesh.field.setNumber(field_id, "XMin", ac - 2*eps)
        gmsh.model.mesh.field.setNumber(field_id, "XMax", L)
        gmsh.model.mesh.field.setNumber(field_id, "YMin",0)
        gmsh.model.mesh.field.setNumber(field_id, "YMax", H/3)
        gmsh.model.mesh.field.setNumber(field_id, "ZMin", -B)
        gmsh.model.mesh.field.setNumber(field_id, "ZMax", 0)


        field2_id = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(field2_id, "VIn", 2 * h)
        gmsh.model.mesh.field.setNumber(field2_id, "VOut", 8 * h)
        gmsh.model.mesh.field.setNumber(field2_id, "XMin", 0)
        gmsh.model.mesh.field.setNumber(field2_id, "XMax", L)
        gmsh.model.mesh.field.setNumber(field2_id, "YMin", 0)
        gmsh.model.mesh.field.setNumber(field2_id, "YMax", H/2)
        gmsh.model.mesh.field.setNumber(field2_id, "ZMin", -B)
        gmsh.model.mesh.field.setNumber(field2_id, "ZMax", 0)


        field3_id = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(field3_id, "VIn", 4* h)
        gmsh.model.mesh.field.setNumber(field3_id, "VOut", 8 * h)
        gmsh.model.mesh.field.setNumber(field3_id, "XMin", 0)
        gmsh.model.mesh.field.setNumber(field3_id, "XMax", L)
        gmsh.model.mesh.field.setNumber(field3_id, "YMin", 0)
        gmsh.model.mesh.field.setNumber(field3_id, "YMax", H)
        gmsh.model.mesh.field.setNumber(field3_id, "ZMin", -B)
        gmsh.model.mesh.field.setNumber(field3_id, "ZMax", 0)



        # Combine the fields
        min_field_id = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field_id, "FieldsList", [field_id, field2_id, field3_id])
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field_id)

        # Generate and optimize the mesh
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Netgen")
        gmsh.write("pureshear.msh")


    model = MPI.COMM_WORLD.bcast(gmsh.model, root=0)
    partitioner = dolfinx.cpp.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet)
    mesh_data = io.gmshio.model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=3, partitioner=partitioner)

    gmsh.finalize()
    MPI.COMM_WORLD.barrier()
    domain = mesh_data[0]
