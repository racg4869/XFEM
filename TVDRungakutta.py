import numpy as np

class TVDRungeKutta:
    def __init__(self,ydot,Z0,dt,t0=0,verbose=1,logger=None):
        self.dt=dt
        self.Z0=Z0
        self.t0=t0
        
        self.Z=Z0
        self.t=t0
        
        self.ydot=ydot
        
        self.verbose=verbose
        self.logger=logger
        
    def iterate(self,dt=None,t=None,Z=None,**kwargs):
        dt=dt if dt else self.dt
        t0=t if t else self.t
        Z0=Z if Z else self.Z
        t,Z,cflbound=self.TVDRungeKutta(t0,Z0,dt,**kwargs)
        
        self.logger.debug('TVD RungeKutta (order=%d) from %f -> %f (CFL step bound: %f)'%(kwargs.get('order',2),t0,t,cflbound))
        if((t-t0)>cflbound):
            self.logger.warning("Warning : dt=(%.3f) > CFL step bound (%.3f) "%(t-t0,cflbound))
        if(np.amax(Z-Z0)<=1e-6*np.amax(Z)):
            self.logger.info("increment very small!")
        
        self.t,self.Z=t,Z
        return self.t,self.Z
    
    def TVDRungeKutta(self,t0,y0,dt,order=2,auto=False,
                      scale=0.9,dtmin=None,dtmax=None):
        """
        refer to :
            Osher S, Shu C W. High-order essentially nonoscillatory schemes for Hamilton-Jacobi equations[J]. SIAM Journal on numerical analysis, 1991, 28(4): 907-922.
            Osher S, Fedkiw R, Piechor K. Level set methods and dynamic implicit surfaces[J]. Appl. Mech. Rev., 2004, 57(3): B15-B15.
                Chapter 3.5
        """
        if auto:
            scale=0.8 if scale is None else scale
            dtmin=1e-5 if dtmin is None else dtmin
            dtmax=1.0 if dtmax is None else dtmax

        RKCoeffs={
            2:[np.array([[1,0],
                         [0.5,0.5]]),
               np.array([[1,0],
                         [0,0.5]]),],
            3:[np.array([[1,0,0],
                         [0.75,0.25,0],
                         [1/3,0,2/3]]),
               np.array([[1,0,0],
                         [0,0.25,0],
                         [0,0,2/3]]),],
            4:[np.array([[1,0,0,0],
                         [0.5,0.5,0,0],
                         [1/9,2/9,2/3,0],
                         [0,1/3,1/3,1/3]]),
               np.array([[0.5,0,0,0],
                         [-0.25,0.5,0,0],
                         [-1/9,-1/3,1,0],
                         [0,1/6,0,1/6]]),],
        }
        
        t,ys,dys=t0,[y0,],[]
        alpha,beta=RKCoeffs[order]
        cflbound=np.infty
        for k in range(order):
            ydot,cflb1=self.ydot(t,ys[k])
            
            cflbound=min(cflbound,cflb1)
            if k==0 and auto:
                dt=cflbound*scale
                dt=max(dt,dtmin)
                dt=min(dt,dtmax)
            
            dys.append(dt*ydot)
            
            t,y=t+dt,np.zeros_like(y0)
            for l in range(k+1):
                y+=alpha[k,l]*ys[l]+beta[k,l]*dys[l]
            ys.append(y)
        
        return t0+dt,ys[-1],cflbound
    
    def ForwardEuler(self,t0,y0,dt):
        ydot,cflbound=self.ydot(t0,y0)
        if(dt>cflbound):
            print("Warning : dt=(%.3f) > CFL step bound (%.3f) "%(dt,cflbound))
        return t0+dt,y0+ydot*dt,cflbound
        
    def reset(self,t=None,Z=None):
        self.Z=self.Z0 if Z is None else Z
        self.t=self.t0 if t is None else t

