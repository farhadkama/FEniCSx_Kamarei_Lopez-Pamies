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


r_cyl = 10 / 2
r_sphere = 18.2

H = 1
z_sphere = H + r_sphere


h = 0.005

make_mesh = True

if make_mesh:
    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
        gmsh.model.add("Pokerchip")

        
        # Step 1: Create the full cylinder
        cyl = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, z_sphere, r_cyl)

        # Step 2: Create the sphere at (0, 0, z_sphere)
        sphere = gmsh.model.occ.addSphere(0, 0, z_sphere, r_sphere)

        # Step 3: Subtract the part of the cylinder that intersects the sphere
        cyl_cut, _ = gmsh.model.occ.cut([(3, cyl)], [(3, sphere)])

        # Step 4: Create a box to extract the quarter cylinder (x = 0, y = 0)
        quarter_box = gmsh.model.occ.addBox(0, 0, -1, r_cyl, r_cyl, z_sphere)

        # Step 5: Intersect the cut cylinder with the box to get the quarter part
        quarter_cyl, _ = gmsh.model.occ.intersect(cyl_cut, [(3, quarter_box)])

        # Step 6: Synchronize to reflect the changes in the model
        gmsh.model.occ.synchronize()

        # Step 7: Add physical group for the resulting quarter cylinder
        group = gmsh.model.addPhysicalGroup(3, [v[1] for v in quarter_cyl])
        gmsh.model.setPhysicalName(3, group, "quarterCylinder")

        # Define mesh size fields
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)


        # Generate and optimize the mesh
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Netgen")
        gmsh.write("poker.msh")


    model = MPI.COMM_WORLD.bcast(gmsh.model, root=0)
    partitioner = dolfinx.cpp.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet)
    mesh_data = io.gmshio.model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=3, partitioner=partitioner)

    gmsh.finalize()
    MPI.COMM_WORLD.barrier()
    domain = mesh_data[0]