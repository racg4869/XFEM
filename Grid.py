# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from constants import VoxelPointOrder,EPS

__all__=['GridHex','GhostCell']

class GridHex:
    """
    >>> GridHex(np.linspce(0,1,11)) # for 1D
    >>> GridHex(np.linspce(0,1,11),np.linspce(0,1,21)) # for 2D
    >>> GridHex(np.linspce(0,1,11),np.linspce(0,1,21),np.linspce(0,1,31)) #for 3D
    """
    def __init__(self,*seeds,chs=2): 
        self.ndim=len(seeds)
        self.shape=tuple(len(seeds[axis]) for axis in range(self.ndim))
        self.seeds=[seeds[axis] for axis in range(self.ndim)]
        
        self.XYZs=np.meshgrid(*seeds,indexing='ij') # little different to meshgrid
        self.X,self.Y,self.Z=[self.XYZs[i] if self.ndim>i else None for i in range(3)]
        self.xs,self.ys,self.zs=[seeds[i] if self.ndim>i else None for i in range(3)]
        
        self.chs=chs
        self.funcs={
            0: self.upwindENO2,
            1: self.upwind,
            2: self.upwindENO2,
            3: self.upwindENO3,
        }

    def mesh_derivate(self):
        X,Y=self.X,self.Y
        m,n=np.shape(X)
        #print(m,n)
        dNjdxi=np.zeros((m-1,n-1,4,2,4))
        weights=np.zeros((m-1,n-1,4))
        for i in range(m-1):
            for j in range(n-1):
                nodes=((i,j),(i,j+1),(i+1,j+1),(i+1,j))
                coords=[(X[k,l],Y[k,l]) for (k,l) in nodes]
                dNjdris=isoparametric2d(coords)
                Js=dNjdris@np.array(coords)
                for k in range(4):
                    ni,nj=nodes[k]
                    J=Js[k,:,:]
                    dNjdxi[i,j,k,:,:]=np.linalg.inv(J)@(dNjdris[k,:,:])
                    weights[i,j,k]=1./np.linalg.det(J)
        return dNjdxi,weights
        
    def grad(self,Z):
        xs,ys=self.xs,self.ys
        fy,fx=np.gradient(Z,ys,xs)
        return np.array([fx,fy])
    
    def grad1(self,Z,mesh_der=None):
        """
        与np.gradient性能有较大差距，但是结果与其类似，不建议使用
        """
        X,Y=self.X,self.Y
        M,N=np.shape(X)
        der,wgh=np.zeros((2,M,N)),np.zeros((M,N))

        dNjdxis,weights=mesh_derivate(xx,yy) if mesh_der is None else mesh_der

        for i in range(M-1):
            for j in range(N-1):
                nodes=((i,j),(i,j+1),(i+1,j+1),(i+1,j))
                fs=[Z[k,l] for k,l in nodes]
                for k in range(4):
                    ni,nj=nodes[k]
                    dNjdxi=dNjdxis[i,j,k,:,:]
                    w=weights[i,j,k]
                    der[:,ni,nj]+=w*(dNjdxi@fs)
                    wgh[ni,nj]+=w
        der=der/wgh
        return der,wgh
    
    def plot(self,Z,title=''):
        if self.ndim==1:
            plt.plot(self.X,Z)
        elif self.ndim==2:
            plt.contourf(self.X,self.Y,Z)
            plt.colorbar()
        elif self.ndim==3:
            raise NotImplementedError("Unsupport 3D plot Now! ")
            pass
        plt.title(title)
        plt.axis('equal')
        return 
    
    def __call__(self,axis,Z):
        return self.funcs[self.chs](axis,Z)
        
    def upwind(self,axis,Z):
        """
        used in upwind scheme.
        At each grid point, define φx− as D−φ and φx+ as D+φ. 
        If u_i > 0, approximate φx with φ−x . 
        If u_i < 0, approximate φx with φ+x .
        """
        xys=self.seeds
        triml1=tuple((slice(None) if i!=axis else slice(1,None) for i in range(ndim)))
        trimr1=tuple((slice(None) if i!=axis else slice(0,-1) for i in range(ndim)))        
        extd=tuple((None if i!=axis else slice(None) for i in range(ndim)))
        
        gc=GhostCell(stencil=1)
        dxy1=gc.periodic(axis,np.diff(xys[axis])[extd])
        
        D0=gc.periodic(axis,Z)
        D1=np.diff(D0,axis=axis)/dxy1

        Dlft,Drht=D1[trimr1],D1[triml1]

        return Dlft,Drht
    
    def upwindENO2(self,axis,Z,subcellfix=False):
        """
        Give second order approximation for One-Sided Derivatives φx− and φx+ 
        there are two possible second order approximations to both the left and right, 
        and  chooses the least oscillatory one according to the 
        
        Refer to:
            1. Osher S, Fedkiw R, Piechor K. Level set methods and dynamic implicit surfaces[J]. Appl. Mech. Rev., 2004, 57(3): B15-B15.
                Chapter 3.3 Hamilton-Jacobi ENO
            2. Mitchell I M. A toolbox of level set methods[J]. UBC Department of Computer Science Technical Report TR-2007-11, 2007.
                upwindFirstENO2.m
                
        Parameters
        __________
        axis : the dimension axis index; 0 means axis 0 , 1 means axis 1
        Z   : function values matrix

        """
        xys,ndim=self.seeds,self.ndim
        
        triml1=tuple((slice(None) if i!=axis else slice(1,None) for i in range(ndim)))
        triml2=tuple((slice(None) if i!=axis else slice(2,None) for i in range(ndim)))
        trimr1=tuple((slice(None) if i!=axis else slice(0,-1) for i in range(ndim)))
        trimr2=tuple((slice(None) if i!=axis else slice(0,-2) for i in range(ndim)))
        trimlr1=tuple((slice(None) if i!=axis else slice(1,-1) for i in range(ndim)))
        extd=tuple((None if i!=axis else slice(None) for i in range(ndim)))
        
        gc2=GhostCell(stencil=2)
        dxy1=gc2.zeroOrderExtrapolation(axis,np.diff(xys[axis])[extd])
        D0=gc2.zeroOrderExtrapolation(axis,Z)#gc2.periodic(axis,Z)
        D1=np.diff(D0,axis=axis)/dxy1
        D2=(D1[triml1]-D1[trimr1])/(dxy1[triml1]+dxy1[trimr1]) # N+2
        
        D1,dxy1=D1[trimlr1],dxy1[trimlr1] # N+1
        D2abs=np.abs(D2) 
        
        # when using second order Newton polynomial interpolation
        #  φ− can be defined from two cases : (i-2,i-1,i) or (i-1,i,i+1)
        #  φ+ can be defined from two cases : (i-1,i,i+1) or (i,i+1,i+2)
        #  According to Ref.1, Choose the one with smaller absolute value which can avoid interpolating near large variations such as discontinuities or steep gradients
        Dlft=D1[trimr1]+dxy1[trimr1]*np.where(D2abs[trimr2]<D2abs[trimlr1],D2[trimr2],D2[trimlr1])
        Drht=D1[triml1]-dxy1[triml1]*np.where(D2abs[trimlr1]<D2abs[triml2],D2[trimlr1],D2[triml2])
        return Dlft,Drht
    
    def upwindENO3(self,axis,Z):
        raise NotImplementedError('Uowind ENO 3 not implemented error!')
    
    def finite_difference(self,axis,Z,index):
        """
        return the first order and second order finite difference of the index of `Z` in the direction of `axis`
        """
        pass
    
    def gradient(self,X,index):
        seeds=self.seeds
        g=np.zeros((self.ndim,))
        for axis in range(self.ndim):
            seed=seeds[axis]
            ind=index[axis]
            f0=X[index]
            if ind==0:
                dx2=seed[ind+1]-seed[ind]
                fr=X[tuple(index[i]+1*(i==axis) for i in range(self.ndim))]
                d=(fr-f0)/dx2
            elif ind==self.shape[axis]-1:
                dx1=seed[ind]-seed[ind-1]
                fl=X[tuple(index[i]-1*(i==axis) for i in range(self.ndim))]
                d=(f0-fl)/dx1
            else:
                dx1=seed[ind]-seed[ind-1]
                fl=X[tuple(index[i]-1*(i==axis) for i in range(self.ndim))]
                dx2=seed[ind+1]-seed[ind]
                fr=X[tuple(index[i]+1*(i==axis) for i in range(self.ndim))]
                d=-(dx2)/(dx1*(dx1 + dx2))*fl+(dx2 - dx1)/(dx1 * dx2)*f0+dx1 / (dx2 * (dx1 + dx2))*fr
            g[axis]=d
        return g
    
    def upwindCoffs(self,ptindex,axis,dirct,order=2):
        """
        返回
        """
        i=ptindex[axis]+dirct*(order+1)
        slc=slice(ptindex[axis],i if i>-1 else None,dirct)
        xs=self.seeds[axis][slc]
        if order==1:
            x0,x1=xs
            dx1=x1-x0
            return  [-1/dx1,1/dx1]
        elif order==2:
            x0,x1,x2=xs
            dx1=x1-x0
            dx2=x2-x0
            return [-1/dx1-1/dx2,1/dx1+1/(dx2-dx1),1/dx2-1/(dx2-dx1)]
        else:
            raise ValueError('order = %d not support'%order)
    
    def upwindValues(self,ptindex,axis,dirct,X,order=2):
        i=ptindex[axis]+dirct*(order+1)
        slc=slice(ptindex[axis],i if i>-1 else None,dirct)
        slc1=tuple(slc if i==axis else ptindex[i] for i in range(self.ndim))
        return X[slc1]

    def isSignChange(self,vindex,X):
        """
        whether if there is sign change in the voxcel
        """
        #考虑浮点数误差
        vals=X[tuple(slice(ind,ind+2) for ind in vindex)].flatten()
        return np.all(np.isfinite(vals)) and np.abs(np.mean(np.where(np.abs(vals)>1e-13,
                                       np.sign(vals),0)))!=1
    def isPoint(self,ptindex):
        """
        返回节点是否是内部节点
        """
        return all(0<=ind<l for l,ind in zip(self.shape,ptindex))

    def coord(self,ptindex):
        return [seed[ind] for seed,ind in zip(self.seeds,ptindex)]

    def neighbours(self,index,order=1):
        """
        返回邻接节点的索引列表
        """
        ndim=self.ndim
        return [tuple((index[k]+j*l*(k==i) for k in range(ndim))) 
                        for i in range(ndim) 
                        for j in (-1,1) 
                        for l in range(1,order+1)
                        if 0<=index[i]+j*l<self.shape[i]]

    def adjacent(self,ptindex,axis,N=1):
        """
        返回 沿axis方向左右两侧与节点index相邻的N个节点的列表的列表
        
        """
        ndim=self.ndim
        
        pts=[[tuple([ptindex[k]-i*(k==axis) for k in range(ndim)]) for i in range(1,N+1)],
             [tuple([ptindex[k]+i*(k==axis) for k in range(ndim)]) for i in range(1,N+1)]]
        return pts

    def locate(self,xs):
        """
        返回xs 所在voxel 的 vindex
        注意 如果xs 不在grid的范围内，则返回最靠近其的边界voxel
        """
        vindex=[0 ]*self.ndim

        for axis in range(self.ndim):
            seed=self.seeds[axis]
            x=xs[axis]
            if x<seed[0]-EPS or x>seed[-1]+EPS:
                #raise ValueError("%s out of grid"%(repr(xs)))
                pass
            
            for i in range(0,len(seed)-1):
                if x<seed[i+1]+EPS:
                    break
            vindex[axis]=i
        return tuple(vindex)

    def delta(self,index,axis):
        """
        计算节点index 沿着axis方向的长度
        """
        seed=self.seeds[axis]
        ind=index[axis] #min(max(index[axis],0),self.grid.X.shape[axis]-2)
        if (ind>=0) and (ind<=len(seed)-2):
            return seed[ind+1]-seed[axis][ind]
        else:
            return (seed[-1]-seed[0])*1E4

    def voxelBound(self,vindex):
        return [(seed[ind],seed[ind+1]) for seed,ind in zip(self.seeds,vindex)]
    
    def voxelValues(self,vindex,X):
        """
        return multi dimension array of values 
        """
        return X[tuple(slice(ind,ind+2) for ind in vindex)]

    def voxelPoints(self,vindex):
        """
        return a list of index of the grid points of voxel `vindex`
        """
        ndim=self.ndim
        return [tuple(vindex[i]+pt[i] for i in range(ndim)) 
                    for pt in VoxelPointOrder[ndim]]
    
class GhostCell:
    """
    Chapter 7 Boundary Conditions and Ghost Cells from 
        LeVeque R J. Finite volume methods for hyperbolic problems[M]. Cambridge university press, 2002.
        
        https://github.com/clawpack/classic/blob/master/src/1d/bc1.f
    >>> X=np.np.linspace(0,23,24).reshape((2,3,4))
    >>> GhostCell(2).zeroOrderExtrapolation(1,X)
    """
    def __init__(self,stencil=1,scale=1.0):
        self.stencil=stencil
        self.scale=scale
        
    def periodic(self,axis,X):
        stencil=self.stencil
        slc1=tuple(slice(None) if i!=axis else slice(-stencil,None) for i in range(X.ndim))
        slc2=tuple(slice(None) if i!=axis else slice(stencil) for i in range(X.ndim))
        return np.concatenate((X[slc1]*self.scale,X,X[slc2]*self.scale),axis=axis)
    
    def zeroOrderExtrapolation(self,axis,X):
        stencil=self.stencil
        slc1=tuple(slice(None) if i!=axis else slice(0,1) for i in range(X.ndim))
        slc2=tuple(slice(None) if i!=axis else slice(-1,None) for i in range(X.ndim))
        
        X0=np.repeat(X[slc1],stencil,axis=axis)*self.scale
        X1=np.repeat(X[slc2],stencil,axis=axis)*self.scale
        
        #print(X0.shape,X1.shape,X.shape)
        return np.concatenate((X0,X,X1),axis=axis)