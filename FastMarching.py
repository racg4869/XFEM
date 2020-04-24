# -*- coding: UTF-8 -*-
import numpy as np
import heapq
from Grid import GridHex
import matplotlib.pyplot as plt
import logging 
from itertools import product
from constants import EPS

fmmlogger=logging.getLogger("FastMarching")
fmmlogger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
fmmlogger.handlers.clear()
fmmlogger.addHandler(ch)

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
        self.logger=logging.getLogger("FastMarching")
        
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
            _,(v,index)=heapq.heappop(self.Tentative)
            if self.Status[index]==0:
                # 由于无法更新非Heap中元素的值，Tentative中存在过期的元素
                if v>vstop:
                    break
                self.n2k(index,v)
                self.updateNeibour(index)
    
    def updateNeibour(self,index):
        """
        updat each  neighbour of `index`
        """
        for neib in self.grid.neighbours(index):
            status=self.Status[neib]
            if status<1:
                # Narrow band  or Far
                t=self.compute(neib)
                if np.isnan(t):
                    self.logger.warning('nan get in compute distance of index=%s'%(repr(neib)))
                elif abs(t)<abs(self.T[neib]):
                    self.f2n(neib,t)

    def n2k(self,ptindex,v):
        self.logger.info('n2k: index=%s,coords=%s,t=%f'%(ptindex,self.grid.coord(ptindex),v))
        self.Status[ptindex]=1
        self.T[ptindex]=v
        self.updateNeibour(ptindex)
    
    def f2n(self,ptindex,v):
        self.logger.info('f2n: index=%s,coords=%s,t=%f,old t=%f'%(repr(ptindex),self.grid.coord(ptindex),v,self.T[ptindex]))
        self.Status[ptindex]=0
        self.T[ptindex]=v
        heapq.heappush(self.Tentative,(abs(v),(v,ptindex)))

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
        dervCoffs,(tmin,tmax)=self.dervCoff(index)
        """
        """
        neibs=[tuple((index[j]+dervCoffs[axis][0]*(j==axis) for j in range(ndim))) 
                         for axis in range(ndim) if dervCoffs[axis][0]!=0]
        neibCoffs=[self.dervCoff(neib)[0] for neib in neibs]
        for axis in range(ndim):
            if dervCoffs[axis][0]!=0:
                dirct,k,b,d1,d2=dervCoffs[axis]
                quadratic[2]=quadratic[2]+k**2
                quadratic[1]=quadratic[1]+2*k*b
                quadratic[0]=quadratic[0]+b**2
            else:
                pass
                # !!!! very important !!! improve the accuracy
                md=max((cof[axis][1]*self.T[neib]+cof[axis][2])**2 for neib,cof in zip(neibs,neibCoffs))
                self.logger.info("difference in axis=%d using %f: neibs=%s;neib coffs=%s;"%(axis,md,repr(neibs),repr(neibCoffs)))
                
                quadratic[0]=quadratic[0]+md
            
        if quadratic[2]!=0:
            c,b,a=quadratic
            sgn=0 if abs((tmin+tmax)/2)<EPS else np.sign((tmin+tmax)/2)
            if b**2-4*a*c<-EPS:
                t= max((-b/k+sgn*abs(1/k) for (dirct,k,b,d1,d2) in dervCoffs if dirct!=0),key=lambda x:abs(x))
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
            
            self.logger.debug("index=%s,derivate=%s,a=%f,b=%f,c=%f,tmin=%f,tmax=%f,t=%f"%(repr(index),repr(dervCoffs),
                                a,b,c,tmin,tmax,t))
            
            # extend the velocity in the front 
            for i in range(len(self.Fext)):
                k,b=0.,0.
                for axis in range(ndim):
                    dirct,kj,bj,d1,d2=dervCoffs[axis]
                    derv=(kj*t+bj)
                    if dirct!=0:
                        v=self.Fext[i][tuple((index[j]+dirct*(j==axis) for j in range(ndim)))]
                        k=k-derv/(dirct*d1)
                        b=b+derv*v/(dirct*d1)
                self.logger.debug("index=%s,d1=%f,k=%f,b=%f,Fext=%f"%(repr(index),d1,k,b,-b/k))
                self.Fext[i][index]=-b/k
            return t
        else:
            self.logger.debug("index=%s,neibs=%s,equation=%s,quadratic=%s"%(repr(index),repr(neibs),repr(dervCoffs),repr(quadratic)))
            raise ValueError('compute invalid point index=%s'%repr(index))
    
    def dervCoff(self,index,order=1):
        """
        返回当前节点index 计算不同方向偏导的差分格式中的系数
        """
        grid=self.grid
        ndim=grid.ndim
        coffes=[]

        coords=grid.coord(index)
        amin,amax=0.0,0.0
        for axis in range(ndim):
            dirct,k,b,d1,d2=0,0,0,np.infty,np.infty
            0<=index[axis]+k<grid.shape[axis]
            for j in (-1,1):
                neib=tuple((index[k]+j*(k==axis) for k in range(ndim)))
                kj,bj,dx1,dx2=0.,np.infty,0.,np.infty
                if (0<=index[axis]+j<grid.shape[axis]) and self.Status[neib]==1:
                    coords1,t1=self.pointInfo(neib)
                    dx1=abs(coords1[axis]-coords[axis])
                    kj,bj=-1/(j*dx1),t1/(j*dx1)
                    amin=min(amin,t1)
                    amax=max(amax,t1)
                    if order==2:
                        neib2=tuple((index[k]+2*j*(k==axis) for k in range(ndim)))
                        if (0<=index[axis]+2*j<grid.shape[axis]) and self.Status[neib2]==1:
                            coords2,t2=self.pointInfo(neib2)
                            dx2=abs(coords2[axis]-coords1[axis])
                            kj=kj-1/(j*(dx1+dx2))
                            bj=bj+t1/(j*dx2)-(dx1/dx2)*(t2/(j*dx1+j*dx2))
                    if (dirct==0) or (bj/kj)>(b/k):
                        dirct,k,b,d1,d2=j,kj,bj,dx1,dx2            
            coffes.append([dirct,k,b,d1,d2])
        return coffes,(amin,amax)
        
    def pointInfo(self,ptindex):
        """
        返回节点index的坐标和T值
        如果节点在外部，则返回边界点的坐标和位置
        """
        #index=tuple(min(max(n,0),self.grid.X.shape[i]-1) for i,n in enumerate(index))
        return self.grid.coord(ptindex),self.T[ptindex]

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
        self.logger=logging.getLogger('Reinitialization')
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

        for vindex in product(*(range(grid.shape[i]-1) for i in range(ndim))):
            if np.any(np.isinf(T0[tuple(slice(ind,ind+2) for ind in vindex)])):
                continue
            if grid.isSignChange(vindex,T0):
                # sign changed
                cof=grid.cubicCoff(vindex,T0)
                for point in grid.voxelPoints(vindex):
                    status[point]=1
                    d=self.cubicDistance(grid,T0,vindex,cof,grid.coord(point))
                    #print(point,d,T0[point],T[point])
                    T[point]=np.sign(T0[point])*min(d,np.abs(T[point]))
                    if extend:
                        # Ref.4 Eq.20
                        x=grid.coord(point)
                        i1,x1,d1=None,None,np.infty
                        i2,x2,d2=None,None,np.infty
                        for i,pt in enumerate(xl):
                            d=sum(((x[j]-pt[j])**2 for j in range(len(x))))**0.5
                            if d<d1:
                                i2,x2,d2=i1,x1,d1
                                i1,x1,d1=i,pt,d
                            elif d<d2:
                                i2,x2,d2=i,pt,d
                        
                        a=sum(((x[j]-x1[j])**2 for j in range(len(x))))
                        b=sum(((x[j]-x2[j])**2 for j in range(len(x))))
                        c=sum(((x[j]-x1[j])*(x[j]-x2[j]) for j in range(len(x))))
                        # (1-alpha)*x1+alpha*x2 is the closet 
                        alpha=min(1,max(0,(a-c)/(a+b-2*c)))
                        for i in range(len(Fext)):
                            Fext[i][point]=(1-alpha)*Fl[i][i1]+alpha*Fl[i][i2]
                        self.logger.debug("point=%s,x=%s,x1=%s,x2=%s,alpha=%f,a=%f,b=%f,c=%f"%(repr(point),repr(x),repr(x1),repr(x2),alpha,a,b,c))
        return status,T,Fext

    def cubicDistance(self,grid,T0,vindex,cof,xs):
        """
        Ref.1 Section 3.2
        """
        ndim=grid.ndim
        
        x0,x=np.array(xs),np.array(xs)
        bnds=grid.voxelBound(vindex)
        vol=np.prod([bnd[1]-bnd[0] for bnd in bnds])
        
        cnt=0
        while True:
            # 是否在外部s
            inside,index1=True,[]
            for i in range(ndim):
                inc=0 
                if x[i]<bnds[i][0]-EPS:
                    inc,inside=-1,False
                elif x[i]>bnds[i][1]+EPS:
                    inc,inside=1,False
                index1.append(vindex[i]+inc)
            
            if inside:
                f,derv=grid.cubicInterpolation(cof,vindex,x)
            else:
                cof1=grid.cubicCoff(index1,T0)
                f,derv=grid.cubicInterpolation(cof1,index1,x)
            
            derv=np.array(derv)
            sqr=np.linalg.norm(derv)**2
            
            dlt1=-f/sqr*derv
            dlt2=(x0-x)-np.dot(x0-x,derv)/sqr*derv
            #print(x0,x,f,derv,dlt1,dlt2)
            
            x=x+dlt1+dlt2
            cnt=cnt+1
            if(np.linalg.norm(dlt1+dlt2)<(1e-3*vol)):
                return np.linalg.norm(x-x0)
            if cnt>20:
                self.logger.warning('cubic distance while loop than 100 times %s -> %s'%(x0,x))
                return np.linalg.norm(x-x0)
    
class ReorthFMM(FastMarching):
    """
    $$
    \nabla \phi \dot \nabla F=0
    $$
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
    
     the front velocity data are provided as a list of sample coordinates, Xl , and the corresponding front velocity vector, Fl.
      search for the two sample coordinates xl1 and xl2 closest to x,
          F_{ijk} =(1−\alpha)F_{l1} +\alpha F_{l2 }
    """
    def __init__(self,grid,phi,xl,Fl):
        self.grid=grid
        self.phi=phi

def test1D():
    grid1D=GridHex(np.linspace(-1,1,101))
    y=grid1D.X**2-0.25
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
    grid2D=GridHex(np.linspace(-1,1,101),np.linspace(-1,1,101))
    phi=grid2D.X**2+grid2D.Y**2-0.25
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
    plt.subplot(2,2,4)
    grid2D.plot(reinit2D.Fext[1])
    plt.title('Fext[1]',fontsize=16)
    plt.axis('equal')  
    plt.show()

    plt.figure(figsize=(12,6))
    plt.suptitle('Error',fontsize=16)
    Z=reinit2D.T
    Z0=np.sqrt(phi+0.25)-np.sqrt(0.25)
    plt.subplot(1,2,1)
    grid2D.plot(np.where(reinit2D.Status>=0,Z-Z0,np.inf),title='Absolute Error of Fast Marching Method')
    plt.axis('equal')
    plt.subplot(1,2,2)
    grid2D.plot(np.where(reinit2D.Status>=0,np.abs((Z-Z0)/Z0),np.inf),title='Relative Error of Fast Marching Method')
    plt.axis('equal')
    plt.show()
    
    print(np.amax(Z[reinit2D.Status==1]))

def test3D():
    grid3D=GridHex(np.linspace(-1,1,21),np.linspace(-1,1,21),np.linspace(-1,1,21))
    phi=grid3D.X**2+grid3D.Y**2+grid3D.Z**2-3*0.5**2
    reinit3D=ReinitializeFMM(grid3D,phi)

def test2D1():
    grid=GridHex(np.linspace(-1,1,101),np.linspace(-1,1,101))
    phi,psi=np.where(grid.X<=-0.1,(grid.X+grid.Y+0.1)/np.sqrt(2),grid.Y),grid.X
    accept=np.abs(phi)<0.2
    fmm=FastMarching(grid,np.where(accept,phi,np.infty),
                    status=np.where(accept,1,-1)) 
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
    grid=GridHex(np.linspace(-1,1,11),np.linspace(-1,1,11))
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

    test2D_line()
    test2D_zx()