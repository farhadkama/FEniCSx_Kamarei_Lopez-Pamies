# Nine circles of elastic brittle fracture: A series of challenge problems to assess fracture models

This repository include the mesh files and codes for nine benchmark fracture problems in FEniCSx to evaluate and validate phase-field models for elastic brittle fracture as described in [1].


## Overview
This repository implements a vetting process in the form of nine challenge problems that any computational model of fracture must convincingly handle if it is to potentially describe fracture nucleation and propagation in general. The focus is on the most basic of settings, that of isotropic elastic brittle materials subjected to quasi-static mechanical loads.

## Challenge Problems

The nine challenge problems have been carefully selected so that:
- They can be carried out experimentally with standard testing equipment
- They can be unambiguously analyzed with a sharp description of fracture
- In aggregate they span the entire range of well settled experimental knowledge on fracture nucleation and propagation that has been amassed for over a century

| Test | Strength Nucleation | Griffith Nucleation | Strength-Griffith Mediated Nucleation | Griffith Propagation Mode I | Griffith Propagation Mode III |
|------|:------------------:|:------------------:|:-------------------------------------:|:---------------------------:|:-----------------------------:|
| **Uniaxial tension** | ✓ | | | | |
| **Biaxial tension** | ✓ | | | | |
| **Torsion** | ✓ | | | | |
| **Pure-shear** | | ✓ | | | |
| **Single edge notch** | | | ✓ | | |
| **Indentation** | | | ✓ | | |
| **Poker-chip** | | | ✓ | | |
| **Double cantilever beam** | | ✓ | | ✓ | |
| **Trousers** | | ✓ | | | ✓ |

## Mesh Files and Example Codes

All mesh files for the nine challenge problems are  available for download through this link:

[**Download All Mesh Files**](https://uofi.box.com/s/4xcgg0syhtniq7lexm21kpne7489gb9e)

The codes for generating these meshes are also available in the `/mesh_generation/` directory, allowing users to modify parameters or create variations as needed.

### Example Codes

The following example implementations are included:

- **Torsion test** - Implementation for linear elasticity problems
- **Trousers test** - Implementation for hyperelasticity problems



##  Usage
This code takes the following **five material properties** as inputs:

1. `E` = Young's modulus  
2. `ν` = Poisson's ratio  
3. `Gc` = Critical energy release rate  
4. `sts` = Tensile strength  
5. `scs` = Compressive strength  

Additionally, the user must specify the **regularization length** `eps` for the boundary value problems. Typically, this length should be chosen so that it is smaller than the smallest size of the structure, as well as the material characteristic length scale $$(3G_c)/(16 W_{ts})$$.

##  Contact

For any inquiry, please contact me at [kamarei2@illinois.edu](mailto:kamarei2@illinois.edu)

Alternatively, you may also reach out to my Ph.D. advisor at [pamies@illinois.edu](mailto:pamies@illinois.edu)


##  References

[1] Kamarei, F., Zeng, B., Dolbow, J.E., Lopez-Pamies, O. (2025). Nine circles of elastic brittle fracture: A series of challenge problems to assess fracture models. Submitted. [PDF](http://pamies.cee.illinois.edu/Publications_files/)


