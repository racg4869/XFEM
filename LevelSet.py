# -*- coding: UTF-8 -*-

import numpy as np
from Grid import GridHex
from TVDRungakutta import TVDRungeKutta
import matplotlib.pyplot as plt
import logging 

logger=logging.getLogger("LevelSet")
logger.handlers.clear()
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

class HamiltonJacobi:
    def __init__(self,grid,Z0,dt):
        self.grid=grid
        self.rk=TVDRungeKutta(self.ydot,Z0,dt,logger=logger)
    
    def ydot(self,t,Z):
        raise NotImplementedError("ydot(t,Z) not implemented yet!")
    
    def iterate(self,**kwargs):
        t,Z = self.rk.iterate(**kwargs)
        return t,Z
    
    def reset(self):
        self.rk.reset()
    
    def steadyState(self,t=None,Z=None,dt=None,tend=None,
                    order=None,nmax=1000,eps=1e-4,
                    auto=False,scale=0.9):
        """
            calculate the steady state of Hamilton-Jacobi Equation 
            until 
                maximum nmax loops 
                or
                norm of ||order not great than eps
        or 
            caculate the solution at t=tend
        
        """
        tend=tend if tend else np.infty
        i,t,Z=0,t if t else self.rk.t0,Z if Z else self.rk.Z0
        while i<nmax:
            i=i+1
            if t<tend:
                t0,Z0=t,Z
                t,Z=self.rk.iterate(dt=dt,auto=auto,scale=scale)
                if(t>tend):
                    self.rk.reset(t0,Z0)
                    t,Z=self.rk.iterate(dt=tend-t0)
                    break
                n1=np.linalg.norm(Z-Z0,ord=order)
                n2=np.linalg.norm(Z0,ord=order)
                logger.debug("t=%f dZ norm %f Z0 norm %f"%(t,n1,n2))                
                if np.abs(n1)<=eps*np.abs(n2):
                    break
            else:
                break
        return t,Z
        
    def plot(self,Z=None,title=None):
        grid=self.grid
        t=0
        if Z is None: 
            Z=self.rk.Z 
            t=self.rk.t
        if title is None: 
            title='t=%.3f\nmin=%.3f\nmax=%.3f'%(t,Z.min(),Z.max())
        grid.plot(Z,title)
        plt.axis('equal')
    
    def solution(self,dt,N,Nstep,NperRow=4,figw=4,**kwargs):
        NRow=np.ceil((N+1)/(Nstep*NperRow))
        fig=plt.figure(figsize=(NperRow*figw,NRow*figw))
        
        plt.subplot(NRow,NperRow,1)
        t,Z=self.rk.t0,self.rk.Z0
        self.plot()
        
        for i in range(N):
            t,Z=self.iterate(dt=dt,**kwargs)
            if i%Nstep==Nstep-1:
                plt.subplot(NRow,NperRow,i//Nstep+2)
                self.plot()
    
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
        
class Reinitialization(HamiltonJacobi):
    """
    0. Mitchell I M. The flexible, extensible and efficient toolbox of level set methods[J]. Journal of Scientific Computing, 2008, 35(2-3): 300-329.
    
    1. Osher S, Fedkiw R, Piechor K. Level set methods and dynamic implicit surfaces[J]. Appl. Mech. Rev., 2004, 57(3): B15-B15.
        Chapter 7. Constructing Signed Distance Functions
    2. Min C. On reinitializing level set functions[J]. Journal of computational physics, 2010, 229(8): 2764-2772.
    
    3. du Chéné A, Min C, Gibou F. Second-order accurate computation of curvatures in a level set framework using novel high-order reinitialization schemes[J]. Journal of Scientific Computing, 2008, 35(2-3): 114-131.
    4. Russo G, Smereka P. A remark on computing distance functions[J]. Journal of computational physics, 2000, 163(1): 51-67.
        Subcell fix Method
    5. Osher S, Shu C W. High-order essentially nonoscillatory schemes for Hamilton–Jacobi equations[J]. SIAM Journal on numerical analysis, 1991, 28(4): 907-922.
        generalized  high order essentially non-oscillatory (ENO) schemes and algorithms of TVD Runge-Kutta type time discretizations 
        
    """
    def __init__(self,grid,Z0,dt,smooth=False,subcellfix=False):
        super().__init__(grid,Z0,dt)

        self.smooth=smooth
        self.subcellfix=subcellfix  # from ref 4,
        
        
    def ydot(self,t,Z,schme="Godunov"):
        """
        Spatial Discretization—Godunov Scheme
        The Godunov Scheme of numerical Hamiltonian
        
        Osher S, Shu C W. High-order essentially nonoscillatory schemes for Hamilton–Jacobi equations[J]. SIAM Journal on numerical analysis, 1991, 28(4): 907-922.
            Chapter 2. SCHEME CONSTRUCTION
        Osher S, Fedkiw R, Piechor K. Level set methods and dynamic implicit surfaces[J]. Appl. Mech. Rev., 2004, 57(3): B15-B15.
            Chapter 5.3.3 
                give the formulation of Godunov’s Scheme

        Min C. On reinitializing level set functions[J]. Journal of computational physics, 2010, 229(8): 2764-2772.
        du Chéné A, Min C, Gibou F. Second-order accurate computation of curvatures in a level set framework using novel high-order reinitialization schemes[J]. Journal of Scientific Computing, 2008, 35(2-3): 114-131.
            give the expression of Godunov Hamiltonian
        """
        grid=self.grid
        dL=max([ np.amin(np.diff(seed)) for seed in grid.seeds])
                
        S=self.smearedSign(Z,dL) if self.smooth else np.sign(Z) 
        mask=(S>0.)
        trimallleft1=tuple((slice(1,None) for axis in range(grid.ndim)))
        
        CFLbound=None
        Hg=np.zeros_like(S)
        for axis in range(grid.ndim):
            deriv_left,deriv_right=self.grid(axis,Z)
            vrightp,vrightn=np.maximum(deriv_right,0),-np.minimum(deriv_right,0)
            vleftp,vleftn=np.maximum(deriv_left,0),-np.minimum(deriv_left,0)
            #print(vrightn,vleftn.shape,vrightp.shape,vleftp.shape)
            vels=mask*np.maximum(vrightn,vleftp)+(~mask)*np.maximum(vrightp,vleftn)
            Hg+=vels**2
            
            extd=tuple((None if i!=axis else slice(None) for i in range(grid.ndim)))
            dx=np.diff(grid.seeds[axis])[extd]
            if CFLbound is None:
                CFLbound=np.abs(vels[trimallleft1]/dx)
            else:
                CFLbound+=np.abs(vels[trimallleft1]/dx)
        
        Hg=S*(1-np.sqrt(Hg))
        cflbound=1/np.amax(CFLbound)
        return Hg,1.0 if np.isinf(cflbound) else cflbound

class TransportEquation(HamiltonJacobi):
    def velocity(self,axis):
        """
        Give the velocity filed of the axis direction
        
        axis : 0,1,2 is x y z resepectively
        
        """
        raise NotImplementedError("velocity(axis) not implemented yet!")
    
    def ydot(self,t,Z,scheme="Godunov"):
        """
        return dy/dt  and CFL step bound of Transport Equation
        """
        grid=self.grid
        
        trimallleft1=tuple((slice(1,None) for axis in range(grid.ndim)))
        trimallright1=tuple((slice(-1) for axis in range(grid.ndim)))
        trimalllr1=tuple((slice(1,-1) for axis in range(grid.ndim)))
        
        CFLbound=None
        Hg=np.zeros_like(Z)
        for axis in range(grid.ndim):
            vel=self.velocity(axis)
            velp,veln=np.maximum(vel,0),np.minimum(vel,0)
            
            deriv_left,deriv_right=self.grid(axis,Z)            
            Hg-=velp*deriv_left+veln*deriv_right
            
            
            trimleft1=tuple((slice(1,None) if i==axis else slice(None) for i in range(grid.ndim)))
            trimright1=tuple((slice(-1) if i==axis else slice(None) for i in range(grid.ndim)))
            extd=      tuple((slice(None) if i==axis else None  for i in range(grid.ndim)))
            dx=np.diff(grid.seeds[axis])[extd]
            
            #print(dx.shape,extd,velp[trimalllr1].shape,dx[trimright1].shape,veln[trimalllr1].shape,dx[trimleft1].shape)
            X=np.abs(velp[trimalllr1]/dx[trimright1]+veln[trimalllr1]/dx[trimleft1])
            if CFLbound is None:
                CFLbound=X
            else:
                CFLbound+=X
        #print(t,np.amax(Hg))
        cflbound=1/np.amax(CFLbound)
        return Hg,1.0 if np.isinf(cflbound) else cflbound
        

class Reorthogonalization(TransportEquation,HamiltonJacobi):
    """
    1. (2001)Burchard P, Cheng L-T, Merriman B, Osher SJ. Motion of curves in three spatial dimensions using a level set approach. Journal of Computational Physics 2001; 170:720–741.
    2. (2002)Non-planar 3D crack growth by the extended 5nite element and level sets—Part II: Level set update
    3. (2006)A study of the representation of cracks with level sets
    
    solve the Hamilton-Jacobi Equation
    
    $$
    \frac{\partial y}{\partial t}+ {\rm sign}(z)\frac{\nabla z}{\mid \nabla z \mid}\dot \nabla y=0
    $$
    
    """
    def __init__(self,grid,Y0,Z0,dt,smooth=True):
        super().__init__(grid,Z0,dt)

        self.Y0=Y0
        self.smooth=smooth

        dL=min([ np.amin(np.diff(seed)) for seed in grid.seeds])
        S=self.smearedSign(Y0,dL) if smooth else np.sign(Y0) 
        
        arr=np.gradient(Y0,*(grid.seeds)) if grid.ndim>1 else (np.gradient(Y0,grid.seeds[0]),)
        grad0=np.stack(arr,axis=0)
        self.normgrad0=S*grad0/np.linalg.norm(grad0,axis=0)[None,]
        self.normgrad0[np.isnan(self.normgrad0)]=0
    
    def velocity(self,axis):
        return self.normgrad0[axis]
    

class LevelSetAdvance(TransportEquation,HamiltonJacobi):
    """
    1. (2001) Gravouil A, Moës N, Belytschko T. Non-planar 3D crack growth by the extended finite element and level sets—Part II: Level set update[J]. International journal for numerical methods in engineering, 2002, 53(11): 2569-2586.
    2. (2006) Marc Duflot. A study of the representation of cracks with level sets
    
    As ref.2 saied,  Update of the Phi function with an advance scalar lead to the inaccuraccy,
    using an advance vector is the better way
    
    grid: grid 
    Z0  : init value
    V   : advance scalar field (must be adjusted to prevent modification of previous crack surface)
    dt  : default time increment
    """
    def __init__(self,grid,Z0,V0,dt):
        super().__init__(grid,Z0,dt)
        
        self.AV=np.stack([gX*V0 for gX in np.gradient(Z0,*(grid.seeds))],
                         axis=0)

    def velocity(self,axis):
        return self.AV[axis]
    
    def steadyState(self,**kwargs):
        kwargs['tend']=1.0
        return super().steadyState(**kwargs)
    
class VelocityExtension(TransportEquation,HamiltonJacobi):
    """
    (1997) Chen S, Merriman B, Osher S, Smereka P. A simple level set method for solving Stefan problems. Journal of Computational Physics 1997; 135:8–29.abs
    (1997) Hou TY, Li Z, Osher S, Zhao H. A hybrid method for moving interface problems with application to the Hele-Shaw Row. Journal of Computational Physics 1997; 134:236–252.    
    (2001) Gravouil A, Moës N, Belytschko T. Non-planar 3D crack growth by the extended finite element and level sets—Part II: Level set update[J]. International journal for numerical methods in engineering, 2002, 53(11): 2569-2586.
    
    """
    def __init__(self,grid,Phis,V0,dt,smooth=False,subcellfix=False):
        super().__init__(grid,V0,dt)
        
    def ydot(self,t,Z,schme="Godunov"):
        """
        """
        pass

class LevelSet:
    def __init__(self,grid,phi0,psi0,dt=0.01,eps=1e-3):
        self.grid=grid
        self.eps=eps
        self.dt=dt
        
        self.phi0=phi0
        self.phi=self.reinitialize(phi0,dt=dt,eps=self.eps).rk.Z
        
        self.psi0=psi0
        psi1=self.reorthogonalize(self.phi,psi0,dt=dt,eps=self.eps).rk.Z
        self.psi=self.reinitialize(psi1,dt=dt,eps=self.eps).rk.Z
        
    def reinitialize(self,Z0,dt=0.01,eps=1e-3):
        rit=Reinitialization(self.grid,Z0,dt)
        rit.steadyState(order=2,eps=eps,auto=True)
        return rit

    def reorthogonalize(self,Y0,Z0,dt=0.01,eps=1e-3):
        rot=Reorthogonalization(self.grid,Y0,Z0,dt)
        rot.steadyState(order=2,eps=eps,auto=True)
        return rot

    def advance(self,Z,V,dt=0.01):
        adv=LevelSetAdvance(self.grid,Z,V,dt)
        adv.steadyState(auto=True)
        return adv
    
    def extend(self,V0,phiOrpsi=0):
        if phiOrpsi==0:
            print(" extend v_phi")
            # extend v_phi
            rot1=self.reorthogonalize(self.phi,V0)
            rot1.steadyState(order=2,eps=1e-3,auto=True)
            
            rot2=self.reorthogonalize(self.psi,rot1.rk.Z)
            rot2.steadyState(order=2,eps=1e-3,auto=True)
        else:
            print(" extend v_psi")
            # extend v_psi
            rot1=self.reorthogonalize(self.psi,V0)
            rot1.steadyState(order=2,eps=1e-3,auto=True)
            rot2=self.reorthogonalize(self.phi,rot1.rk.Z)
            rot2.steadyState(order=2,eps=1e-3,auto=True)
        return rot2.rk.Z
    
    def plotCrack(self):
        pass
    
    def update(self,vphi0,vpsi0):
        print(" extend vphi and vpsi to the domain")
        # extend vphi and vpsi to the domain
        vphi=self.extend(vphi0,phiOrpsi=0)
        self.grid.plot(vphi)
        plt.show()
        
        vpsi=self.extend(vpsi0,phiOrpsi=1)
        self.grid.plot(vpsi)
        plt.show()
        print(" adjustment to prevent modi5cation of previous crack surface")
        # adjustment to prevent modi5cation of previous crack surface
        #vphi=vphi/vpsi*np.maximum(psi,0)
        #vphi[np.isnan(vphi)]=0.0
        
        print(" update the phi level set")
        # update the phi level set
        adv=self.advance(self.phi,vphi)
        self.phi=adv.rk.Z
        adv.plot()
        plt.show()
        
        print(" reinitialize the phi level set")
        # reinitialize the phi level set
        rit_phi=self.reinitialize(self.phi)
        self.phi=rit_phi.rk.Z
        rit_phi.plot()
        plt.show()
        
        print(" update the psi level set")
        # update the psi level set
        adv=self.advance(self.psi,vpsi)
        self.psi=adv.rk.Z
        adv.plot()
        plt.show()
        
        print(" orthogonalize the psi level set")
        # orthogonalize and reinitialize the level se
        rot=self.reorthogonalize(self.phi,self.psi)
        rot.plot()
        plt.show()
        
        self.grid.plot(rot.rk.Z)
        self.grid.plot(self.phi)
        plt.show()
        print(" reinitialize the psi level set")
        rit_psi=self.reinitialize(rot.rk.Z)
        self.psi=rit_psi.rk.Z
        rit_psi.plot()
        plt.show()
        
        return vphi,vpsi,self.phi,self.psi