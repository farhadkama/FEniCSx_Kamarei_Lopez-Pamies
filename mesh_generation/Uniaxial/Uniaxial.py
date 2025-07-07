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

L = 15
r_cyl = 2

eps = 0.2
h = 0.03

make_mesh = True

if make_mesh:
    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
        gmsh.model.add("Uniaxial")
      
        # Step 1: Create the full cylinder
        inner_cyl = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, L, r_cyl)

        # Step 2: Create box to cut out one-quarter (x >= 0 and y >= 0)
        box = gmsh.model.occ.addBox(0, 0, -1, r_cyl, r_cyl, L + 2)

        # Step 3: Intersect cylinder with the quarter-space box
        quarter_cyl, _ = gmsh.model.occ.intersect([(3, inner_cyl)], [(3, box)])

        ## Synchronize to reflect the changes in the model
        gmsh.model.occ.synchronize()

        # Add physical group for the volume (the tube itself)
        group = gmsh.model.addPhysicalGroup(3, [v[1] for v in quarter_cyl])
        gmsh.model.setPhysicalName(3, group, "quarterCylinder")


        # Define mesh size fields
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)

        # Generate and optimize the mesh
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Netgen")
        gmsh.write("Uni.msh")


    model = MPI.COMM_WORLD.bcast(gmsh.model, root=0)
    partitioner = dolfinx.cpp.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet)
    mesh_data = io.gmshio.model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=3, partitioner=partitioner)

    gmsh.finalize()
    MPI.COMM_WORLD.barrier()
    domain = mesh_data[0]