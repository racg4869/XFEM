# -*- coding: UTF-8 -*-

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from itertools import product
import logging
from constants import EPS

from Grid import GridHex
from FMM import FastMarching,ReinitializeFMM,isSignChange,closestTwo
from cubic import CubicInterpolate
from constants import logger,ITMAX

def minmod(a,b):
    return a if abs(a)<=abs(b) else b
    
class FMM_3D(FastMarching):
    """
    1. Bærentzen J A. On the implementation of fast marching methods for 3D lattices[J]. 2001.
    2. Sukumar N, Chopp D L, Moran B. Extended finite element method and fast marching method for three-dimensional fatigue crack propagation[J]. Engineering Fracture Mechanics, 2003, 70(1): 29-48.
    3. Sethian J A. Fast marching methods[J]. SIAM review, 1999, 41(2): 199-235.    
    4. Sukumar N, Chopp D L, Béchet E, et al. Three‐dimensional non‐planar crack growth by a coupled extended finite element and fast marching method[J]. International journal for numerical methods in engineering, 2008, 76(5): 727-748.
        Important
    5. Chopp D L. Some improvements of the fast marching method[J]. SIAM Journal on Scientific Computing, 2001, 23(1): 230-244.
        Give the detail for 2D case
    6. Shi J, Chopp D, Lua J, et al. Abaqus implementation of extended finite element method using a level set representation for three-dimensional fatigue crack growth and life predictions[J]. Engineering Fracture Mechanics, 2010, 77(14): 2840-2863.
        Give the detail for 3D case
    7. Adalsteinsson D , Sethian J A . The Fast Construction of Extension Velocities in Level Set Methods[J]. Journal of Computational Physics, 1999, 148(1):2-22.
    8. Jovičić G R, Živković M, Jovičić N. Numerical modeling of crack growth using the level set fast marching method[J]. FME Transactions, 2005, 33(1): 11-19.

    """
    def __init__(self,grid,phi0,psi0,extend=None):
        self.logger=logger
        
        self.grid=grid
        self.phi0=phi0
        self.phi=phi0.copy()
        self.psi0=psi0
        self.psi=psi0.copy()

        self.phi_psi_cof=[dict(),dict()]
        """
        # used in Reintialization to 
        if extend is None:
            xl,Fl=[],[]
        else:
            xl,Fl=extend['xl'],extend['Fl']

        status,rho,Fext=self.frontInitialize(grid,phi0,psi0,xl,Fl)
        super().__init__(grid,rho,status,V=1.0,Fext=Fext)
        """
    
    def extendVelocity(self,xl,Fl,vstop):
        self.logger.info("Start Compute the distance to the crack front and  Extend the tip velocity")
        grid=self.grid
        phi,psi=self.phi,self.psi
        
        status,rho,Fext=self.frontInitialize(grid,phi,psi,xl,Fl)
        reinit=FastMarching(grid,rho,status,V=1.0,Fext=Fext)

        if vstop is None:
            vstop=8*max(np.max(seed[1:]-seed[:-1]) for seed in grid.seeds)
        reinit.loop(vstop=vstop)
        
        self.logger.info("End! ")
        
        rho,Fext=reinit.T,reinit.Fext
        return rho,Fext
    
    def advanceFront1(self,xl,Fl,dt=1.0,vstop=None):
        grid=self.grid

        rho,Fext=self.extendVelocity(xl,Fl,vstop=vstop)
        phi_n,psi_n=self.phi,self.psi

        grad_phi=np.gradient(phi_n,*grid.seeds)
        grad_psi=np.gradient(psi_n,*grid.seeds)

        phi_n=phi_n-dt*(grad_phi[0]*Fext[0]+grad_phi[1]*Fext[1]+grad_phi[2]*Fext[2])
        psi_n=psi_n-dt*(grad_psi[0]*Fext[0]+grad_psi[1]*Fext[1]+grad_psi[2]*Fext[2])

        reinit1=ReinitializeFMM(grid,psi_n)
        reinit1.loop(vstop=vstop)
        return rho,Fext,phi_n,reinit1.T

    def advanceFront(self,xl,Fl,dt=1.0,vstop=None):
        """
        Three‐dimensional non‐planar crack growth by a coupled extended finite element and fast marching method
            Chapter 3.1.3
        """
        grid=self.grid
        phi_n,psi_n=self.phi,self.psi

        rho,Fext=self.extendVelocity(xl,Fl,vstop=vstop)
        grad_rho=np.gradient(rho,*grid.seeds)
        grad_phi=np.gradient(phi_n,*grid.seeds)

        norm_rho=np.sqrt(grad_rho[0]*grad_rho[0]+grad_rho[1]*grad_rho[1]+grad_rho[2]*grad_rho[2])
        norm_F=np.sqrt(Fext[0]*Fext[0]+Fext[1]*Fext[1]+Fext[2]*Fext[2])

        psi_n_=rho*(grad_rho[0]*Fext[0]+grad_rho[1]*Fext[1]+grad_rho[2]*Fext[2])/(norm_rho*norm_F)

        phi_n1=phi_n.copy()
        for ind in product(*[range(x) for x in rho.shape]):
            p1,p2,p2_1,r=phi_n[ind],psi_n[ind],psi_n_[ind],rho[ind]
            if p2_1<=0:
                if p2>0:
                    phi_n1[ind]=np.sign(p1)*r
            else:
                F=np.array([Fext[i][ind] for i in range(3)])
                g_phi=np.array([grad_phi[i][ind] for i in range(3)])
                g_rho=np.array([grad_rho[i][ind] for i in range(3)])

                v=np.cross(np.cross(F,g_phi),F)        
                x=np.dot(g_rho,v)*(r/(norm_rho[ind]*np.linalg.norm(v)))
                
                if p2<=0:
                    phi_n1[ind]=minmod(p1,x)
                else:
                    phi_n1[ind]=x

        #grad_psi_=np.gradient(psi_n_,*grid.seeds)
        #norm_F=grad_psi_[0]*Fext[0]+grad_psi_[1]*Fext[1]+grad_psi_[2]*Fext[2]
        psi_n1=psi_n_-norm_F*dt
        reinit1=ReinitializeFMM(grid,psi_n1)
        reinit1.loop(vstop=vstop)
        return rho,Fext,phi_n1,reinit1.T

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
        self.logger.info("3D Front Initialize Start")
        ndim=grid.ndim
        
        status=np.full(grid.shape,-1)
        rho=np.full(grid.shape,np.inf)
        Fext=[np.full(grid.shape,np.inf) for _ in range(len(Fl))]
        
        extend=(len(Fl)>0)
        cub_phi=CubicInterpolate(grid,phi)
        cub_psi=CubicInterpolate(grid,psi)

        for vindex in product(*[range(grid.shape[i]-1) for i in range(ndim)]):
            val_phi=grid.voxelValues(vindex,phi)
            val_psi=grid.voxelValues(vindex,psi)
            if isSignChange(val_phi) and isSignChange(val_psi):
                self.logger.debug("voxel %s : phi=%s; psi=%s"%(repr(vindex),repr(val_phi),repr(val_psi)))
                for point in grid.voxelPoints(vindex):
                    crd=grid.coord(point)
                    d=self.cubicDistance(cub_phi,cub_psi,crd)
                    if not np.isfinite(d):
                        self.logger.warning("voxel %s cubic distance is wrong!"%(repr(vindex)))
                        break
                    status[point]=1
                    rho[point]=min(d,rho[point])
                    if extend:
                        i1,i2,alpha=closestTwo(xl,crd)
                        for i in range(len(Fext)):
                            Fext[i][point]=(1-alpha)*Fl[i][i1]+alpha*Fl[i][i2]
        self.logger.info("3D Front Initialize End")
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

class Crack3D_Evolution(FastMarching):
    pass

def test3D_line():
    grid=GridHex(np.linspace(-1,1,21),np.linspace(-1,1,21),np.linspace(-1,1,2))
    phi0=grid.Y-0.2
    psi0=grid.X-0.0

    xl=[(0,0.2,x) for x in np.linspace(-1,1,21)]
    Fl=[[0.1 for x in xl],
        [0.1 for x in xl],
        [0.0 for x in xl]]

    vstop=1.0 # np.infty #

    crack=FMM_3D(grid,phi0,psi0)
    rho,Fext,phi_n1,psi_n1=crack.advanceFront(xl,Fl,dt=1.0,vstop=vstop)

    slc=(slice(None),slice(None),1)
    grad_rho=np.gradient(rho,*grid.seeds)
    Xs=[
        rho,rho-np.sqrt(crack.phi**2+crack.psi**2),
        crack.phi,crack.psi,
        phi_n1,psi_n1
    ]
    subsize=(int((len(Xs)+1)/2),2)
    plt.figure(figsize=tuple(7*x for x in subsize[-1::-1]))
    for i,X in enumerate(Xs):
        plt.subplot(*subsize,i+1)
        plt.contourf(grid.X[slc],grid.Y[slc],X[slc])
        plt.axis('equal')
        plt.colorbar()
    plt.show()

def test3D_circle():
    grid=GridHex(np.linspace(-1,1,51),np.linspace(-1,1,51),np.linspace(-1,1,2))
    phi0=grid.Y-0.2
    psi0=grid.X-0.0

    xl=[(0,0.2,x) for x in np.linspace(-1,1,51)]
    Fl=[[0.2 for x in xl],
        [0.2 for x in xl],
        [0.0 for x in xl]]

    vstop=np.infty #0.4 # 

    crack=FMM_3D(grid,phi0,psi0)
    rho,Fext,phi_n1,psi_n1=crack.advanceFront(xl,Fl,dt=1.0,vstop=vstop)

    slc=(slice(None),slice(None),1)
    grad_rho=np.gradient(rho,*grid.seeds)
    Xs=[#rho,*grad_rho,
        crack.phi,crack.psi,
        phi_n1,psi_n1
    ]
    subsize=(int((len(Xs)+1)/2),2)
    plt.figure(figsize=tuple(7*x for x in subsize[-1::-1]))
    for i,X in enumerate(Xs):
        plt.subplot(*subsize,i+1)
        plt.contourf(grid.X[slc],grid.Y[slc],X[slc])
        plt.axis('equal')
        plt.colorbar()
    plt.show()

if __name__ == "__main__":
    test3D_line()