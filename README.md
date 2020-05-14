# XFEM

## Motion of surface

1. Osher S, Sethian J A. Fronts propagating with curvature-dependent speed: algorithms based on Hamilton-Jacobi formulations[J]. Journal of computational physics, 1988, 79(1): 12-49.
2. Sethian J A. Level set methods and fast marching methods: evolving interfaces in computational geometry, fluid mechanics, computer vision, and materials science[M]. Cambridge university press, 1999.
    > **Very Impotant**

XFEM coupled with a level set representation  of the crack  has shown to be a powerful and efficient tool for crack propagation simulations

In the level set representation, the crack is usually described by one( for 2D ) or two (for 3D) zero-level sets of functions.
In 3D case, the two level set function are called normal and tangential level set.
So the crack in the Finite Element Model can be introduced independently of the mesh.

There are two numerical technique (Level Set Method and Fast Marching Method) for tracking the evolution of interfaces. 


### Level Set Method
+ Hou T Y, Li Z, Osher S, et al. A hybrid method for moving interface problems with application to the Hele–Shaw flow[J]. Journal of Computational Physics, 1997, 134(2): 236-252.
+ Peng D, Merriman B, Osher S, et al. A PDE-based fast local level set method[J]. Journal of computational physics, 1999, 155(2): 410-438.
+ Stolarska M, Chopp D L, Moës N, et al. Modelling crack growth by level sets in the extended finite element method[J]. International journal for numerical methods in Engineering, 2001, 51(8): 943-960.
    > **Impotant**
+ Moës N, Gravouil A, Belytschko T. Non‐planar 3D crack growth by the extended finite element and level sets—Part I: Mechanical model[J]. International journal for numerical methods in engineering, 2002, 53(11): 2549-2568.
+ Gravouil A, Moës N, Belytschko T. Non‐planar 3D crack growth by the extended finite element and level sets—Part II: Level set update[J]. International journal for numerical methods in engineering, 2002, 53(11): 2569-2586.
+ Osher S, Fedkiw R, Piechor K. Level set methods and dynamic implicit surfaces[J]. Appl. Mech. Rev., 2004, 57(3): B15-B15.
    > **Impotant**
+ Duflot M. A study of the representation of cracks with level sets[J]. International journal for numerical methods in engineering, 2007, 70(11): 1261-1302.
    > **Impotant**
+ Colombo D, Massin P. Fast and robust level set update for 3D non-planar X-FEM crack propagation modelling[J]. Computer methods in applied mechanics and engineering, 2011, 200(25-28): 2160-2180.
+ 

### Fast Marching Method
+ Sethian J A. A fast marching level set method for monotonically advancing fronts[J]. Proceedings of the National Academy of Sciences, 1996, 93(4): 1591-1595.
+ Adalsteinsson D, Sethian J A. The fast construction of extension velocities in level set methods[J]. Journal of Computational Physics, 1999, 148(1): 2-22.
+ Sethian J A. Fast marching methods[J]. SIAM review, 1999, 41(2): 199-235.
    > **Impotant**
+ Chopp D L. Some improvements of the fast marching method[J]. SIAM Journal on Scientific Computing, 2001, 23(1): 230-244.
+ Sukumar N, Chopp D L, Moran B. Extended finite element method and fast marching method for three-dimensional fatigue crack propagation[J]. Engineering Fracture Mechanics, 2003, 70(1): 29-48.
+ Jovičić G R, Živković M, Jovičić N. Numerical modeling of crack growth using the level set fast marching method[J]. FME Transactions, 2005, 33(1): 11-19.
+ Sukumar N, Chopp D L, Béchet E, et al. Three‐dimensional non‐planar crack growth by a coupled extended finite element and fast marching method[J]. International journal for numerical methods in engineering, 2008, 76(5): 727-748.
    > **Impotant**
+ Shi J, Chopp D, Lua J, et al. Abaqus implementation of extended finite element method using a level set representation for three-dimensional fatigue crack growth and life predictions[J]. Engineering Fracture Mechanics, 2010, 77(14): 2840-2863.
+ Alblas D. Implementing and analysing the fast marching method[D]. University of Twente, 2018.

FMM is computationally attractive for monotonically advancing fronts, was first introduced by Sethian, and later improved by Sethian and Chopp.

## XFEM

1. Moës N, Dolbow J, Belytschko T. A finite element method for crack growth without remeshing[J]. International journal for numerical methods in engineering, 1999, 46(1): 131-150.


In the X-FEM, a discontinuous function and the two-dimensional asymptotic crack-tip displacement fields are added to the finite element
approximation to account for the crack using the notion of partition of unity