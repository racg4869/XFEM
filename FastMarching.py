# -*- coding: UTF-8 -*-
import numpy as np
import heapq
from Grid import GridHex
import matplotlib.pyplot as plt
import logging 
from itertools import product
from constants import EPS
from scipy import optimize

from cubic import CubicInterpolate
from constants import logger,ITMAX
from closestTwo import closestTwo

class FastMarching:
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
    
    Fast Marching Method
        Ref 1 give the detail about 

    Reinitialization

    Velocity Extension
        Ref.4 Section 3.1.2
        Ref.5 Section 3.3
     the front velocity data are provided as a list of sample coordinates, `xl` , 
     and the corresponding front velocity vector, `Fl`.
     search for the two sample coordinates xl1 and xl2 closest to x,
          F_{ijk} =(1−\alpha)F_{l1} +\alpha F_{l2 }
    
    """
    def __init__(self,grid,T0,status,V=None,Fext=[]):
        self.logger=logger
        
        self.grid=grid
        self.T0=T0
        self.T=np.where(status==1,T0,np.inf) #np.sign(T0)*
        
        # Velocity 
        self.V=V
        # Extend Field
        self.Fext=Fext

        # status of each grid points : 
        #   -1 for Distant : also know as Far, far from the initial interface to be possible candidates for 
        #   0 for Tentaive : also know as Narrow Band, all potential candidates to be the next mesh point to be added to the set A
        #   1 for Accepted : also know as Frozen, All the mesh points are considered computed and are always closer to the initial interface than any of the remaining mesh points
        self.Status=status
        
        # all potential candidates to be the next mesh point to be added to the set A
        #  kept sorted in a heap sort
        self.Tentative=[]
        heapq.heapify(self.Tentative)
        
        self.initialize()
            
    def velocity(self,index):
        if self.V is None:
            raise NotImplementedError('No Velocity !')
        else:
            return self.V if np.isscalar(self.V) else self.V[index]

    def initialize(self):
        for index in np.argwhere(self.Status==1):
            index=tuple(index)
            self.updateNeibour(index)
    
    def loop(self,vstop=np.infty):
        """
        loop update procedure until reach `vstop`
        """
        while(len(self.Tentative)>0):
            _,(v,Fext,index)=heapq.heappop(self.Tentative)
            if self.Status[index]==0:
                # 由于无法更新非Heap中元素的值，Tentative中存在过期的元素
                if v>vstop:
                    break
                self.n2k(index,v,Fext)
                self.updateNeibour(index)
    
    def updateNeibour(self,index):
        """
        updat each  neighbour of `index`
        """
        for neib in self.grid.neighbours(index):
            status=self.Status[neib]
            if status<1:
                # Narrow band  or Far
                t,Fext=self.compute(neib)
                self.f2n(neib,t,Fext)

    def n2k(self,ptindex,v,Fext):
        self.Status[ptindex]=1
        self.T[ptindex]=v
        for i in range(len(self.Fext)):
            self.Fext[i][ptindex]=Fext[i]
        self.logger.info('n2k: index=%s,coords=%s,t=%f,Fext=%s'%(ptindex,self.grid.coord(ptindex),v,repr(Fext)))
        self.updateNeibour(ptindex)
    
    def f2n(self,ptindex,v,Fext):
        if np.isnan(v):
            self.logger.warning('nan get in compute distance of index=%s'%(repr(ptindex)))
        elif abs(v)<abs(self.T[ptindex]):
            self.logger.info('f2n: index=%s,coords=%s,t=%f->%f,Fext=%s'%(repr(ptindex),self.grid.coord(ptindex),self.T[ptindex],v,repr(Fext)))
            self.Status[ptindex]=0
            self.T[ptindex]=v
            heapq.heappush(self.Tentative,(abs(v),(v,Fext,ptindex)))

    def compute(self,index):
        """
        compute the distance of the voxel 
        Ref.1. Appendix A: Pseudo Code
        Ref.2. Section 3: Fast Marching Method (7a),(7b)
        Ref.3. Section 5:  Higher-Accuracy Fast Marching Methods
        """
        grid=self.grid
        ndim=grid.ndim
        
        if self.Status[index]==1:
            return 
        
        quadratic=[-1/self.velocity(index),0,0] # c,b,a for ax^2+bx+c=0
        
        # list of (+1 for right -1 for left, phi_1, delta_x)
        dervCoffs,(tmin,tmax)=self.upwindCoff(index)
        neibs=None
        neibCoffs=None
        for axis in range(ndim):
            if dervCoffs[axis][0]!=0:
                dirct,k,b,cfs=dervCoffs[axis]
                quadratic[2]=quadratic[2]+k**2
                quadratic[1]=quadratic[1]+2*k*b
                quadratic[0]=quadratic[0]+b**2
            else:
                pass
                """
                if neibs==None:
                    neibs=[tuple((index[j]+dervCoffs[axis][0]*(j==axis) for j in range(ndim))) 
                                for axis in range(ndim) if dervCoffs[axis][0]!=0 ]
                    neibCoffs= [self.upwindCoff(neib)[0] for neib in neibs if grid.isPoint(neib)]
                
                # !!!! very important !!! improve the accuracy
                md=max((cof[axis][1]*self.T[neib]+cof[axis][2])**2 for neib,cof in zip(neibs,neibCoffs))
                self.logger.info("difference in axis=%d using %f: neibs=%s;neib coffs=%s;"%(axis,md,repr(neibs),repr(neibCoffs)))
                quadratic[0]=quadratic[0]+md
                """

        if quadratic[2]!=0:
            c,b,a=quadratic
            sgn=0 if abs((tmin+tmax)/2)<EPS else np.sign((tmin+tmax)/2)
            if b**2-4*a*c<-EPS:
                t= max((-b/k+sgn*abs(1/k) for (dirct,k,b,_) in dervCoffs if dirct!=0),key=lambda x:abs(x))
            elif abs(b**2-4*a*c)<EPS:
                t=-b/(2*a)
            else:
                q=0.5*(-b-(1 if b>=0 else -1)*np.sqrt(b**2-4*a*c))
                t1,t2=max(q/a,c/q),min(q/a,c/q)
                if t1*t2>0:
                    # 同号选择绝对值大的那个作为解
                    t=t1 if abs(t1)>abs(t2) else t2  
                else:
                    if abs(abs(t1)-abs(t2))<EPS:
                        t=sgn*abs(t1)
                    elif np.sign(t1)==sgn:
                        t=t1
                    elif np.sign(t2)==sgn:
                        t=t2
                    else:
                        self.logger.error("index=%s,a=%f,b=%f,c=%f,tmin=%f,tmax=%f,t1=%f,t2=%f"%(repr(index),a,b,c,tmin,tmax,t1,t2))
                        raise ValueError('compute invalid point index=%s'%repr(index))
            
            self.logger.debug("index=%s,t=%f,derivate=%s,a=%f,b=%f,c=%f,tmin=%f,tmax=%f"%(repr(index),t,repr(dervCoffs),
                                a,b,c,tmin,tmax))
            
            # extend the velocity in the front
            Fext=[0.,]*len(self.Fext)
            for i in range(len(self.Fext)):
                k,b=0.,0.
                for axis in range(ndim):
                    dirct,kj,bj,cfs=dervCoffs[axis]
                    derv=kj*t+bj
                    if dirct!=0:
                        k=k+derv*cfs[0]
                        b=b+derv*np.dot(cfs[1:],
                            grid.upwindValues(index,axis,dirct,self.Fext[i],order=len(cfs)-1)[1:])

                self.logger.debug("index=%s,k=%f,b=%f,Fext=%f"%(repr(index),k,b,-b/k))
                if abs(b/k)<1e6:
                    Fext[i]=-b/k
            return t,Fext
        else:
            self.logger.warn("index=%s,equation=%s,quadratic=%s"%(repr(index),repr(dervCoffs),repr(quadratic)))
            raise ValueError('compute invalid point index=%s'%repr(index))
    
    def upwindCoff(self,index,order=2):
        """
        返回当前节点index 计算不同方向偏导的差分格式中的系数
        """
        grid=self.grid
        ndim=grid.ndim
        coffes=[]

        amin,amax=np.infty,-np.infty
        for axis in range(ndim):
            dirct,k,b,cfs=0,0,0,[]
            for j in (-1,1):
                neib=tuple((index[k]+j*(k==axis) for k in range(ndim)))
                neib2=tuple((index[k]+2*j*(k==axis) for k in range(ndim)))
                val=[]
                if (0<=neib[axis]<grid.shape[axis]) and self.Status[neib]==1:
                    val.append(self.T[neib])
                    if (0<=neib2[axis]<grid.shape[axis]) and self.Status[neib2]==1:
                        val.append(self.T[neib2])
                    cfs=grid.upwindCoffs(index,axis,j,order=len(val))
                    amin=min(amin,min(val))
                    amax=max(amax,max(val))
                    kj,bj=cfs[0],np.dot(cfs[1:],val)
                    #kj,bj=-j*kj,-j*bj # gudonov scheme max(f^-,-f^+,0)
                    if (dirct==0) or abs(bj/kj)>abs(b/k):
                        # 等长
                        dirct,k,b=j,kj,bj
            coffes.append([dirct,k,b,cfs])
        return coffes,(amin,amax)
        
class ReinitializeFMM(FastMarching):
    """
    1. Chopp D L. Some improvements of the fast marching method[J]. SIAM Journal on Scientific Computing, 2001, 23(1): 230-244.
        Give the detail for 2D case
    2. Shi J, Chopp D, Lua J, et al. Abaqus implementation of extended finite element method using a level set representation for three-dimensional fatigue crack growth and life predictions[J]. Engineering Fracture Mechanics, 2010, 77(14): 2840-2863.
        Give the detail for 3D case
    3. Sukumar N, Chopp D L, Béchet E, et al. Three‐dimensional non‐planar crack growth by a coupled extended finite element and fast marching method[J]. International journal for numerical methods in engineering, 2008, 76(5): 727-748.
    4. Adalsteinsson D , Sethian J A . The Fast Construction of Extension Velocities in Level Set Methods[J]. Journal of Computational Physics, 1999, 148(1):2-22.
    
    Reinitialization 
        
    Tip Velocity Extension
        Ref.1 Section 3.3
        Ref.3 Section 3.1.2
        Ref.4 
     the front velocity data are provided as a list of sample coordinates, Xl , and the corresponding front velocity vector, Fl.
      search for the two sample coordinates xl1 and xl2 closest to x,
          F_{ijk} =(1−\alpha)F_{l1} +\alpha F_{l2 }    
    """
    def __init__(self,grid,X,extend=None):
        if extend is None:
            xl,Fl=[],[]
        else:
            xl,Fl=extend['xl'],extend['Fl']
        self.logger=logger
        status,X,Fext=self.frontInitialize(grid,X,xl,Fl)

        super().__init__(grid,X,status,V=1.0,Fext=Fext)
        
    def frontInitialize(self,grid,T0,xl=[],Fl=[]):
        """
        reconstruct process provides local tricubic approximation 
        of the initial conditions on grid points near the crack front
        """
        ndim=grid.ndim
        
        status=np.full_like(T0,-1)
        T=np.full(grid.shape,np.inf)

        extend=len(Fl)>0
        Fext=[np.full(grid.shape,np.inf) for _ in range(len(Fl))]
        
        cub_T0=CubicInterpolate(grid,T0)
        
        for vindex in product(*(range(grid.shape[i]-1) for i in range(ndim))):
            if np.any(np.isinf(T0[tuple(slice(ind,ind+2) for ind in vindex)])):
                continue
            if grid.isSignChange(vindex,T0):
                # sign changed
                for point in grid.voxelPoints(vindex):
                    status[point]=1
                    #print(grid.coord(point))
                    d=self.cubicDistance(cub_T0,grid.coord(point))
                    #print(point,d,T0[point],T[point])
                    T[point]=np.sign(T0[point])*min(d,np.abs(T[point]))
                    if extend:
                        # Ref.4 Eq.20
                        x=grid.coord(point)
                        i1,i2,alpha=closestTwo(xl,x)
                        x1,x2=xl[i1],xl[i2]
                        for i in range(len(Fext)):
                            Fext[i][point]=(1-alpha)*Fl[i][i1]+alpha*Fl[i][i2]
                        self.logger.debug("point=%s,x=%s,x1=%s,x2=%s,alpha=%f"%(repr(point),repr(x),repr(x1),repr(x2),alpha))
        return status,T,Fext

    def cubicDistance(self,cub_T0:CubicInterpolate,xs):
        """
        Ref.1 Section 3.2
        """
        grid=cub_T0.grid
        ndim=grid.ndim

        vindex=grid.locate(xs)
        bnds=grid.voxelBound(vindex)
        vol=np.prod([bnd[1]-bnd[0] for bnd in bnds])

        x0,x=np.array(xs),np.array(xs)
        cnt=0
        """
        # using `Conjugate Gradient Method` to solve minimum distance
        def func(x):
            xs=np.array(x[:-1])
            a=x[-1]
            p=cub_T0.interpolate(xs)[0]
            return np.dot(x0-xs,x0-xs)+a**2*p**2
        def jac(x):
            xs=np.array(x[:-1])
            a=x[-1]
            dr=2*(xs-x0)
            p,dp=cub_T0.interpolate(xs)
            return np.concatenate([dr+(2*a**2*p)*np.asarray(dp),[2*a*p**2]])
        
        res=optimize.minimize(fun=func,jac=jac,method="CG",
            x0=np.concatenate([x0,[0]]))
        print(x0,res)
        return np.sqrt(res.fun)
        """

        while True:
            f,derv=cub_T0.interpolate(x)
            derv=np.array(derv)
            sqr=np.linalg.norm(derv)**2
            dlt1=-f/sqr*derv
            dlt2=0.0 #(x0-x)-np.dot(x0-x,derv)/sqr*derv
            dx=dlt1+dlt2
            x=x+dx
            cnt=cnt+1
            if(np.linalg.norm(dx)<(1e-3*vol)):
                return np.linalg.norm(x-x0)
            if cnt>ITMAX:
                self.logger.warning('cubic distance loop %d times %s -> %s'%(ITMAX,x0,x))
                return np.linalg.norm(x-x0)
        
class ReorthFMM(FastMarching):
    """
    can be used to 
     1. extended velocity `F` on the grid points near the crack front into the rest of the domain
        in such a way that F is constant in the direction normal to the interface
     2. 
    """
    def __init__(self,grid,phi,F0,status):
        super().__init__(grid,F0)
        
        
    def compute(self,ptindex):
        pass
    
class CrackEvolution(FastMarching):
    """
    
    """
    def __init__(self,grid,phi0,psi0):
        self.Phi=np.zeros_like(grid.X)
        self.Psi=np.zeros_like(grid.X)
        
        # Evaluate the front speed F at n discrete points on the front.  
        # determine the speed on grid point near front according to 
    def initialize(self):
        """
        1.  Initialize all the points adjacent to the initial interface with an initial value, put those points in A. 
        2.  All points X i,j,k not ∈ A,but are adjacent to a point in A are given initial estimates for X i,j,k by solving Equation (3) 
            for the given configuration of neighboring points in A. These points are tentative points and put in the set T.
        3, All remaining points are placed in D and given initial value of ? i,j,k =+∞.
        """
    def estimate(self):
        """
         estimate of phi for points in T
        """

class ExtendingVelocity(FastMarching):
    """
    1. Chopp D L. Some improvements of the fast marching method[J]. SIAM Journal on Scientific Computing, 2001, 23(1): 230-244.
        Give the detail for 2D case
    2. Shi J, Chopp D, Lua J, et al. Abaqus implementation of extended finite element method using a level set representation for three-dimensional fatigue crack growth and life predictions[J]. Engineering Fracture Mechanics, 2010, 77(14): 2840-2863.
        Give the detail for 3D case
    3. Sukumar N, Chopp D L, Béchet E, et al. Three‐dimensional non‐planar crack growth by a coupled extended finite element and fast marching method[J]. International journal for numerical methods in engineering, 2008, 76(5): 727-748.    
    4. Adalsteinsson D , Sethian J A . The Fast Construction of Extension Velocities in Level Set Methods[J]. Journal of Computational Physics, 1999, 148(1):2-22.
    
    Ref.1 Section 3.3
    Ref.3 Section 3.1.2
    
    xl
    $$
    \nabla \phi \dot \nabla F=0
    $$

    initial 
     the front velocity data are provided as a list of sample coordinates, Xl , and the corresponding front velocity vector, Fl.
      search for the two sample coordinates xl1 and xl2 closest to x,
          F_{ijk} =(1−\alpha)F_{l1} +\alpha F_{l2 }
    
    solving

    """
    def __init__(self,grid,phi,xl,Fl):
        self.grid=grid
        self.phi=phi

        self.xl=xl
        self.Fl=Fl
        self.Fext=[np.zeros(self.grid.shape) for _  in range(len(Fl))]

    def frontInitialize(self,T0):
        xl,Fl=self.xl,self.Fl
        grid=self.grid

        ndim=grid.ndim
        
        status=np.full_like(T0,-1)
        T=np.full(grid.shape,np.inf)

        extend=len(Fl)>0
        Fext=[np.full(grid.shape,np.inf) for _ in range(len(Fl))]

        for vindex in product(*(range(grid.shape[i]-1) for i in range(ndim))):
            if np.any(np.isinf(T0[tuple(slice(ind,ind+2) for ind in vindex)])):
                continue
            if grid.isSignChange(vindex,T0):
                # sign changed
                for point in grid.voxelPoints(vindex):
                    status[point]=1
                    # Ref.4 Eq.20
                    x=grid.coord(point)
                    
                    i1,i2,alpha=closestTwo(xl,x)
                    x1,x2=xl[i1],xl[i2]
                    for i in range(len(Fext)):
                        Fext[i][point]=(1-alpha)*Fl[i][i1]+alpha*Fl[i][i2]
                    self.logger.debug("point=%s,x=%s,x1=%s,x2=%s,alpha=%f"%(repr(point),repr(x),repr(x1),repr(x2),alpha))

    def upwind(self):
        pass
    
    def loop(self,vstop=np.infty):
        pass
def test1D():
    grid1D=GridHex(np.linspace(-1,1,101))
    y=grid1D.X**2-0.25
    exact=np.where(grid1D.X>0,grid1D.X-0.5,grid1D.X+0.5)
    reinit1D=ReinitializeFMM(grid1D,y,extend={"xl":[(-0.5,),(0.5,)],"Fl":[[-1,1],]})
    #reinit1D.Fext=[grid1D.X.copy(),]
    grid1D.plot(reinit1D.T)
    plt.show()

    reinit1D.loop(vstop=0.5)

    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    grid1D.plot(reinit1D.T)
    plt.title('End',fontsize=16)
    plt.axis('equal')
    plt.subplot(1,2,2)
    grid1D.plot(reinit1D.Fext[0])
    plt.title('',fontsize=16)
    plt.axis('equal')
    plt.show()

def test2D():
    #  2D
    grid2D=GridHex(np.linspace(-1,1,51),np.linspace(-1,1,51))
    phi=grid2D.X**2+grid2D.Y**2-0.25
    exact=np.sqrt(phi+0.25)-0.5

    xl=[(0.5*np.cos(theta),0.5*np.sin(theta)) for theta in np.linspace(0,2*np.pi,101)]
    Fl=[[x for x,y in xl],[y for x,y in xl]]
    reinit2D=ReinitializeFMM(grid2D,phi,extend={"xl":xl,"Fl":Fl})
    #reinit2D.Fext=[reinit2D.grid.X.copy(),reinit2D.grid.Y.copy()]

    plt.figure(figsize=(15,12))
    plt.suptitle('Initial',fontsize=16)
    plt.subplot(2,2,1)
    grid2D.plot(reinit2D.T0)
    plt.title('T0',fontsize=16)
    plt.axis('equal')
    plt.subplot(2,2,2)
    grid2D.plot(reinit2D.T)
    plt.title('T',fontsize=16)
    plt.axis('equal')
    plt.subplot(2,2,3)
    grid2D.plot(reinit2D.Fext[0])
    plt.title('Fext[0]',fontsize=16)
    plt.axis('equal')
    plt.subplot(2,2,4)
    grid2D.plot(reinit2D.Fext[1])
    plt.title('Fext[1]',fontsize=16)
    plt.axis('equal')
    plt.show()

    reinit2D.loop(vstop=1.0)

    plt.figure(figsize=(15,12))
    plt.suptitle('End',fontsize=16)
    plt.subplot(2,2,1)
    grid2D.plot(reinit2D.T0)
    plt.title('T0',fontsize=16)
    plt.axis('equal')
    plt.subplot(2,2,2)
    grid2D.plot(reinit2D.T)
    plt.title('T',fontsize=16)
    plt.axis('equal')
    plt.subplot(2,2,3)
    grid2D.plot(reinit2D.Fext[0])
    plt.title('Fext[0]',fontsize=16)
    plt.axis('equal')
    plt.clim(-0.5,0.5)
    plt.subplot(2,2,4)
    grid2D.plot(reinit2D.Fext[1])
    plt.title('Fext[1]',fontsize=16)
    plt.axis('equal')
    plt.show()

    plt.figure(figsize=(12,6))
    plt.suptitle('Error',fontsize=16)
    Z=reinit2D.T
    Z0=exact
    plt.subplot(1,2,1)
    grid2D.plot(np.where(reinit2D.Status>=0,Z-Z0,np.inf),title='Absolute Error of Fast Marching Method')
    plt.axis('equal')
    plt.subplot(1,2,2)
    grid2D.plot(np.where(reinit2D.Status>=0,np.abs((Z-Z0)/Z0),np.inf),title='Relative Error of Fast Marching Method')
    plt.axis('equal')
    plt.show()
    
    print(np.amax(Z[reinit2D.Status==1]))

def test2D_1():
    grid2d=GridHex(np.linspace(-1,1,41),np.linspace(-1,1,41))
    phi=grid2d.X**2+grid2d.Y**2
    xl=[(0,0),(0,0)]
    Fl=[(0,0),(1,1)]
    reinit2D=ReinitializeFMM(grid2d,phi,extend={"xl":xl,"Fl":Fl})

def test2D1():
    grid=GridHex(np.linspace(-1,1,101),np.linspace(-1,1,101))
    phi,psi=np.where(grid.X<=-0.1,(grid.X+grid.Y+0.1)/np.sqrt(2),grid.Y),grid.X
    accept=np.abs(phi)<0.2
    fmm=FastMarching(grid,np.where(accept,phi,np.infty),
                    status=np.where(accept,1,-1)) 

def test3D():
    grid3D=GridHex(np.linspace(-1,1,21),np.linspace(-1,1,21),np.linspace(-1,1,21))
    phi=grid3D.X**2+grid3D.Y**2+grid3D.Z**2-3*0.5**2
    reinit3D=ReinitializeFMM(grid3D,phi)

def test2D_xx():
    Nx,Ny=101,101
    grid=GridHex(np.linspace(-1,1,Nx),np.linspace(-1,1,Ny))
    index=(0,0)
    X=np.full_like(grid.X,np.inf)
    status=np.full((Nx,Ny),-1,dtype=np.int8)

    theta=np.pi/4
    cet=np.array([(seed[ind]+seed[ind+1])/2 for seed,ind in zip(grid.seeds,index)])
    for i,j in product(*[range(4) for _ in range(2)]):
        ind1,ind2=index[0]+i,index[1]+j
        status[ind1,ind2]=1
        crd=np.array([seed[ind] for seed,ind in zip(grid.seeds,(ind1,ind2))])
        X[ind1,ind2]=np.dot(crd-cet,np.array([np.cos(theta),np.sin(theta)]))

    fmm=ReinitializeFMM(grid,X)
    fmm.loop(vstop=1.0)
    grid.plot(fmm.T)
    plt.axis('equal')
    plt.show()

def test2D_line():
    grid=GridHex(np.linspace(-1,1,11),np.linspace(-1,1,11))
    exact=grid.Y-0.3
    accept=np.logical_and(np.abs(exact)<=(2*2/(11-1)+1e-6),grid.X<0.3)
    fmm=FastMarching(grid,np.where(accept,exact,np.infty),status=accept,V=1.0)
    fmm.loop(vstop=0.5)
    fmm.grid.plot(fmm.T)
    plt.axis('equal')
    plt.show()

def test2D_zx():
    grid=GridHex(np.linspace(-1,1,51),np.linspace(-1,1,51))
    phi,psi=np.where(grid.X<=-0.1,(grid.X+grid.Y+0.1)/np.sqrt(2),grid.Y),grid.X
    accept=np.abs(phi)<0.2
    fmm=ReinitializeFMM(grid,phi)
    fmm.loop(vstop=1.0)
    grid.plot(fmm.T)
    plt.show()

if __name__=="__main__":
    #grid2D=GridHex(np.linspace(-1,1,101),np.linspace(-1,1,101))
    #phi=grid2D.X**2+grid2D.Y**2-0.25
    #xl=[(0.5*np.cos(theta),0.5*np.sin(theta)) for theta in np.linspace(0,2*np.pi,101)]
    #Fl=[[x for x,y in xl],[y for x,y in xl]]
    #reinit2D=ReinitializeFMM(grid2D,phi,extend={"xl":xl,"Fl":Fl})
    #reinit2D.loop(vstop=1.0)
    
    #test1D()

    test2D()
    #test2D1()
    #test2D_1()
    
    #test2D_line()
    #test2D_zx()

    #test2D_xx()