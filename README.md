# Nine circles of elastic brittle fracture: A series of challenge problems to assess fracture models

This repository includes the mesh files, the scripts to generate the meshes, and the FEniCSx FE code to solve the nine challenge fracture problems described in [1].

## Overview
The nine challenge problems introduced in [1] aim at establishing a minimum standard or vetting process — within the simplest setting of isotropic elastic brittle materials subjected to quasi-static mechanical loads — that any computational model of fracture ought to pass, if it is to potentially describe fracture nucleation and propagation in general.

## Challenge Problems
The table below lists the nine challenge problems alongside the type of fracture nucleation and/or propagation that they characterize. If a model fails to deliver accurate predictions for one of these problems, then such a model is not a viable candidate to describe — and hence predict — fracture in general.   


Critically, the problems are such that:
- They can be carried out experimentally with standard testing equipment
- They can be unambiguously analyzed with a sharp description of fracture
- In aggregate they span the entire range of well settled experimental knowledge on fracture nucleation and propagation that has been amassed for over a century.

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
- **Torsion test**
- **Trousers test**



##  Usage
The code for the linear elastic brittle material (soda-lime glass) takes the following **five material constants** as inputs:

1. `E` = Young's modulus  
2. `ν` = Poisson's ratio  
3. `Gc` = Critical energy release rate  
4. `sts` = Uniaxial tensile strength 
5. `shs` = Hydrostatic strength

The code for the non-linear elastic (Neo-Hookean) brittle material (a PU elastomer) takes the following **five material constants** as inputs:
1. `mu` = Shear modulus  
2. `lambda` = Lame constant 
3. `Gc` = Critical energy release rate  
4. `sts` = Uniaxial tensile strength  
5. `shs` = Hydrostatic strength

Additionally, the user must specify the **regularization length** `eps` for the boundary value problems. Typically, this length should be chosen so that it is smaller than the smallest size of the structure, as well as the material characteristic length scale $$(3G_c)/(16 W_{ts})$$.

##  Contact

For any inquiry, please contact me at [kamarei2@illinois.edu](mailto:kamarei2@illinois.edu)

Alternatively, you may also reach out to my Ph.D. advisor at [pamies@illinois.edu](mailto:pamies@illinois.edu)


##  References

[1] Kamarei, F., Zeng, B., Dolbow, J.E., Lopez-Pamies, O. (2025). *Nine circles of elastic brittle fracture: A series of challenge problems to assess fracture models*. Submitted. [PDF](https://arxiv.org/pdf/2507.00266)


