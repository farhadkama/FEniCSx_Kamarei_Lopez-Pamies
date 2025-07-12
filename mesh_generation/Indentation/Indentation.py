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


# Parameters for the outer rectangle
ac = 1
L = 25
B = 25


eps = 0.016
h = 0.005

make_mesh = True

if make_mesh:
    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
        gmsh.model.add("AxisymmetricIndentation")

        
        # Step 1: Create rectangle (2D surface = 2D entity)
        rect = gmsh.model.occ.addRectangle(0, 0, 0, B, -L)


        # Step 2: Synchronize to update the CAD model
        gmsh.model.occ.synchronize()

        # Step 3: Add physical group for the resulting domain
        group = gmsh.model.addPhysicalGroup(2, [rect])
        gmsh.model.setPhysicalName(2, group, "Rectangle")

        # Define mesh size fields
        field_id = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(field_id, "VIn", h)
        gmsh.model.mesh.field.setNumber(field_id, "VOut", 32*h)
        gmsh.model.mesh.field.setNumber(field_id, "XMin", 0)
        gmsh.model.mesh.field.setNumber(field_id, "XMax", ac*5)
        gmsh.model.mesh.field.setNumber(field_id, "YMin", -ac*5)
        gmsh.model.mesh.field.setNumber(field_id, "YMax", 0)




        field2_id = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(field2_id, "VIn", 4*h)
        gmsh.model.mesh.field.setNumber(field2_id, "VOut",  32*h)
        gmsh.model.mesh.field.setNumber(field2_id, "XMin", 0)
        gmsh.model.mesh.field.setNumber(field2_id, "XMax", ac*10)
        gmsh.model.mesh.field.setNumber(field2_id, "YMin", -ac*8)
        gmsh.model.mesh.field.setNumber(field2_id, "YMax", 0)




        field3_id = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(field3_id, "VIn",  8*h)
        gmsh.model.mesh.field.setNumber(field3_id, "VOut",  32*h)
        gmsh.model.mesh.field.setNumber(field3_id, "XMin", 0)
        gmsh.model.mesh.field.setNumber(field3_id, "XMax", ac*15)
        gmsh.model.mesh.field.setNumber(field3_id, "YMin", -ac*10)
        gmsh.model.mesh.field.setNumber(field3_id, "YMax", 0)




        # Combine the fields
        min_field_id = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field_id, "FieldsList", [field_id, field2_id, field3_id])
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field_id)

        # Generate and optimize the mesh
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Netgen")
        gmsh.write("Inden2D.msh")


    model = MPI.COMM_WORLD.bcast(gmsh.model, root=0)
    partitioner = dolfinx.cpp.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet)
    mesh_data = io.gmshio.model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=2, partitioner=partitioner)

    gmsh.finalize()
    MPI.COMM_WORLD.barrier()
    domain = mesh_data[0]