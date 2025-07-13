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


L1 = 1.5
L2 = 50
L = L1 + L2

H = 10
B = 2.5/2

x_cyl = L1
y_cyl = H/3
z_cyl = B
r_cyl = B / 4

ac = 25
ac += L1


eps = 0.16
h = 0.05

make_mesh = True

if make_mesh:
    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
        gmsh.model.add("DCB")


        # Create outer box
        block = gmsh.model.occ.addBox(0, 0, 0, L, H, -B)

        # Create inner cylinder (same axis, smaller radius)
        inner_cyl = gmsh.model.occ.addCylinder(x_cyl, y_cyl, 0, 0, 0, -z_cyl, r_cyl)

        # Cut inner cylinder from outer cylinder to form a tube
        dcb, _ = gmsh.model.occ.cut([(3, block)], [(3, inner_cyl)])

        # Synchronize to reflect the changes in the model
        gmsh.model.occ.synchronize()

        # Add physical group for the volume (the tube itself)
        dcb_volumes = [entity[1] for entity in dcb]
        dcb_group = gmsh.model.addPhysicalGroup(3, dcb_volumes)
        gmsh.model.setPhysicalName(3, dcb_group, "dcbVolume")


        # Define mesh size fields
        field_id = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(field_id, "VIn", h)
        gmsh.model.mesh.field.setNumber(field_id, "VOut", 2 * h)
        gmsh.model.mesh.field.setNumber(field_id, "XMin", ac - 2*eps)
        gmsh.model.mesh.field.setNumber(field_id, "XMax", L)
        gmsh.model.mesh.field.setNumber(field_id, "YMin",0)
        gmsh.model.mesh.field.setNumber(field_id, "YMax", H/3)
        gmsh.model.mesh.field.setNumber(field_id, "ZMin", -B)
        gmsh.model.mesh.field.setNumber(field_id, "ZMax", 0)


        field2_id = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(field2_id, "VIn", 2 * h)
        gmsh.model.mesh.field.setNumber(field2_id, "VOut", 4 * h)
        gmsh.model.mesh.field.setNumber(field2_id, "XMin", 0)
        gmsh.model.mesh.field.setNumber(field2_id, "XMax", L)
        gmsh.model.mesh.field.setNumber(field2_id, "YMin", 0)
        gmsh.model.mesh.field.setNumber(field2_id, "YMax", 2*H/3)
        gmsh.model.mesh.field.setNumber(field2_id, "ZMin", -B)
        gmsh.model.mesh.field.setNumber(field2_id, "ZMax", 0)


        field3_id = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(field3_id, "VIn", 2* h)
        gmsh.model.mesh.field.setNumber(field3_id, "VOut", 4 * h)
        gmsh.model.mesh.field.setNumber(field3_id, "XMin", x_cyl - 1.1*r_cyl)
        gmsh.model.mesh.field.setNumber(field3_id, "XMax", x_cyl + 1.1*r_cyl)
        gmsh.model.mesh.field.setNumber(field3_id, "YMin", y_cyl - 1.1*r_cyl)
        gmsh.model.mesh.field.setNumber(field3_id, "YMax", y_cyl + 1.1*r_cyl)
        gmsh.model.mesh.field.setNumber(field3_id, "ZMin", -B)
        gmsh.model.mesh.field.setNumber(field3_id, "ZMax", 0)



        # Combine the fields
        min_field_id = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field_id, "FieldsList", [field_id, field2_id, field3_id])
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field_id)

        # Generate and optimize the mesh
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Netgen")
        gmsh.write("dcb.msh")


    model = MPI.COMM_WORLD.bcast(gmsh.model, root=0)
    partitioner = dolfinx.cpp.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet)
    mesh_data = io.gmshio.model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=3, partitioner=partitioner)

    gmsh.finalize()
    MPI.COMM_WORLD.barrier()
    domain = mesh_data[0]

