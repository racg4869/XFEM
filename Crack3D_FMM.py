# -*- coding: UTF-8 -*-

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from itertools import product
import logging
from constants import EPS

from Grid import GridHex
from FastMarching import FastMarching,ReinitializeFMM
from cubic import CubicInterpolate
from constants import logger,ITMAX
from closestTwo import closestTwo

class Crack3D_FMM(FastMarching):
    """
    1. Bærentzen J A. On the implementation of fast marching methods for 3D lattices[J]. 2001.
    2. Sukumar N, Chopp D L, Moran B. Extended finite element method and fast marching method for three-dimensional fatigue crack propagation[J]. Engineering Fracture Mechanics, 2003, 70(1): 29-48.
    3. Sethian J A. Fast marching methods[J]. SIAM review, 1999, 41(2): 199-235.    
    4. Sukumar N, Chopp D L, Bechet E, et al. Three‐dimensional non‐planar crack growth by a coupled extended finite element and fast marching method[J]. International Journal for Numerical Methods in Engineering, 2008, 76(5): 727-748.
    5. Chopp D L. Some improvements of the fast marching method[J]. SIAM Journal on Scientific Computing, 2001, 23(1): 230-244.
        Give the detail for 2D case
    6. Shi J, Chopp D, Lua J, et al. Abaqus implementation of extended finite element method using a level set representation for three-dimensional fatigue crack growth and life predictions[J]. Engineering Fracture Mechanics, 2010, 77(14): 2840-2863.
        Give the detail for 3D case
    7. Adalsteinsson D , Sethian J A . The Fast Construction of Extension Velocities in Level Set Methods[J]. Journal of Computational Physics, 1999, 148(1):2-22.
    """
    def __init__(self,grid,phi0,psi0,extend=None):
        self.logger=logger
        
        self.grid=grid
        self.phi0=phi0
        self.phi=phi0.copy()
        self.psi0=psi0
        self.psi=psi0.copy()

        # used in Reintialization to 
        if extend is None:
            xl,Fl=[],[]
        else:
            xl,Fl=extend['xl'],extend['Fl']

        self.phi_psi_cof=[dict(),dict()]

        status,rho,Fext=self.frontInitialize(grid,phi0,psi0,xl,Fl)
        super().__init__(grid,rho,status,V=1.0,Fext=Fext)

    def advanceFront(self,xl,Fl):
        grid=self.grid
        phi,psi=self.phi,self.psi
        
        status,rho,Fext=self.frontInitialize(grid,phi,psi,xl,Fl)
        
        self.logger.info("Start Compute the distance to the crack front and  Extend the tip velocity")
        reinit=FastMarching(grid,rho,status,V=1.0,Fext=Fext)
        vstop=6*max(np.max(seed[1:]-seed[:-1]) for seed in grid.seeds)
        reinit.loop(vstop=vstop)
        self.logger.info("End! ")
        
        rho,Fext=self.reinit.T,self.reinit.Fext
        
        grad_rho=np.gradient(rho,grid.seeds)
        normG=np.sqrt(grad_rho[0]*grad_rho[0]+grad_rho[1]*grad_rho[1]+grad_rho[2]*grad_rho[2])
        normF=np.sqrt(Fext[0]*Fext[0]+Fext[1]*Fext[1]+Fext[2]*Fext[2])
        
        psi1=rho*(grad_rho[0]*Fext[0]+grad_rho[1]*Fext[1]+grad_rho[2]*Fext[2])/(normG*normF)
        
        phi1=np.where(np.logical_and(psi<=0,psi1<=0),phi,)
        
    def frontInitialize(self,grid,phi,psi,xl,Fl):
        """
        compute 
            1. the distance to the crack front using tricubic approximation 
            2. the velocity vector interpolate from the velocity vector `Fl` at crack front `xl`
        for those grid points near the crack front (cut by crack front)

        Tip Velocity Extension
            Ref.1 Section 3.3
            Ref.3 Section 3.1.2
            Ref.4 
        the front velocity data are provided as a list of sample coordinates, Xl , and the corresponding front velocity vector, Fl.
          search for the two sample coordinates xl1 and xl2 closest to x,
          F_{ijk} =(1−\alpha)F_{l1} +\alpha F_{l2 }            

        """
        ndim=grid.ndim
        
        status=np.full(grid.shape,-1)
        rho=np.full(grid.shape,np.inf)
        Fext=[np.full(grid.shape,np.inf) for _ in range(len(Fl))]
        
        extend=(len(Fl)>0)
        cub_phi=CubicInterpolate(grid,phi)
        cub_psi=CubicInterpolate(grid,psi)

        for vindex in product(*[range(grid.shape[i]-1) for i in range(ndim)]):
            if grid.isSignChange(vindex,phi) and grid.isSignChange(vindex,psi):
                for point in grid.voxelPoints(vindex):
                    status[point]=1
                    crd=grid.coord(point)
                    d=self.cubicDistance(cub_phi,cub_psi,crd)
                    rho[point]=min(d,rho[point])
                    if extend:
                        i1,i2,alpha=closestTwo(xl,crd)
                        x1,x2=xl[i1],xl[i2]
                        for i in range(len(Fext)):
                            Fext[i][point]=(1-alpha)*Fl[i][i1]+alpha*Fl[i][i2]
                        self.logger.debug("index=%s,crd=%s,x1=%s,x2=%s,alpha=%f"%(repr(point),repr(crd),repr(x1),repr(x2),alpha))

        return status,rho,Fext
        

    def cubicDistance(self,cub_phi,cub_psi,xs):
        """
        Modified Newton Method for cubic interpolation 
        introduced in 

        """
        grid=self.grid
        ndim=grid.ndim
        vindex=grid.locate(xs)
        bnds=grid.voxelBound(vindex)
        vol=np.prod([bnd[1]-bnd[0] for bnd in bnds])
        
        x0,x=np.array(xs),np.array(xs)
        cnt=0
        while True:
            p,derv_p=cub_phi.interpolate(x)
            q,derv_q=cub_psi.interpolate(x)
            derv_p=np.array(derv_p)
            derv_q=np.array(derv_q)
            
            derv=p*derv_p+q*derv_q
            sqr=np.dot(derv,derv)
            
            dlt1=-(p**2+q**2)/sqr*derv
            dlt1[np.isnan(dlt1)]=0.
            
            derv=np.cross(derv_p,derv_q)
            sqr=np.dot(derv,derv)
            dlt2=0.0 #np.dot(x0-x,derv)/sqr*derv
            #dlt2[np.isnan(dlt2)]=0.
            
            x=x+dlt1+dlt2
            cnt=cnt+1
            #self.logger.debug("x=%s;p=%f;q=%f;derv_p=%s;derv_q=%s;dlt1=%s;dlt2=%s"%(repr(x),p,q,repr(derv_p),repr(derv_q),repr(dlt1),repr(dlt2)))
            if(np.linalg.norm(dlt1+dlt2)<(1e-3*vol)):
                return np.linalg.norm(x-x0)
            if cnt>ITMAX:
                self.logger.warning('cubic distance loop %d times %s -> %s'%(ITMAX,x0,x))
                return np.linalg.norm(x-x0)
        
    @staticmethod
    def smearedSign(Z,sgnfactor):
        return Z/np.sqrt(Z**2+sgnfactor**2)

    @staticmethod
    def minabs(A,B):
        """
        (2007)Antoine du Chéné, Chohong Min, Frédéric Gibou. Second-Order Accurate Computation of Curvatures in a Level Set Framework Using Novel High-Order Reinitialization Schemes(2007)Antoine du Chéné, Chohong Min, Frédéric Gibou. Second-Order Accurate Computation of Curvatures in a Level Set Framework Using Novel High-Order Reinitialization Schemes
        >>> minabs(-1,-2),minabs(1,-2),minabs(-1,2),minabs(1,2)
        """
        return np.where(abs(A)<abs(B),A,B)

    @staticmethod
    def minmod(A,B):
        """
        (2007)Antoine du Chéné, Chohong Min, Frédéric Gibou. Second-Order Accurate Computation of Curvatures in a Level Set Framework Using Novel High-Order Reinitialization Schemes
        >>> minmod(np.linspace(0,1,11),-0.5),minmod(np.linspace(0,1,11),0.5)
        """
        mask1=(A*B)>0
        mask2=abs(A)>abs(B)
        return B*np.logical_and(mask1,mask2)+A*np.logical_and(mask1,np.logical_not(mask2))

def test3D():
    grid=GridHex(np.linspace(-1,1,21),np.linspace(-1,1,21),np.linspace(-1,1,3))
    phi0=grid.Y-0.3
    psi0=grid.X-0.0
    xl=[(0,0.3,x) for x in np.linspace(-1,1,21)]
    Fl=[[1.0 for x in xl],
        [2.0 for x in xl],
        [x[2] for x in xl]]

    crack=Crack3D_FMM(grid,phi0,psi0,extend={"xl":xl,"Fl":Fl})

    vstop=np.inf #6*max(np.max(seed[1:]-seed[:-1]) for seed in grid.seeds)
    crack.loop(vstop=vstop)
    rho,Fext=crack.T,crack.Fext

    grad_rho=np.gradient(rho,*grid.seeds)

    normG=np.sqrt(grad_rho[0]*grad_rho[0]+grad_rho[1]*grad_rho[1]+grad_rho[2]*grad_rho[2])
    normF=np.sqrt(Fext[0]*Fext[0]+Fext[1]*Fext[1]+Fext[2]*Fext[2])
    phi2=rho*(grad_rho[0]*Fext[0]+grad_rho[1]*Fext[1]+grad_rho[2]*Fext[2])/(normG*normF)

    slc=(slice(None),slice(None),0)
    plt.figure(figsize=(14,6))
    plt.subplot('221')
    plt.contourf(grid.X[slc],grid.Y[slc],rho[slc])
    plt.axis('equal')
    plt.colorbar()
    plt.subplot('222')
    plt.contourf(grid.X[slc],grid.Y[slc],phi2[slc])
    plt.axis('equal')
    plt.colorbar()
    slc=(slice(None),slice(None),1)
    plt.subplot('223')
    plt.contourf(grid.X[slc],grid.Y[slc],rho[slc])
    plt.axis('equal')
    plt.colorbar()
    plt.subplot('224')
    plt.contourf(grid.X[slc],grid.Y[slc],phi2[slc])
    plt.axis('equal')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    test3D()