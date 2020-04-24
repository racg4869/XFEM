# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from functools import partial

VoxelPointOrder={
    1:[(0,),(1,)],
    2:[(0,0),(0,1),(1,0),(1,1)],
    3:[(0, 0, 0),(0, 0, 1),(0, 1, 0),(0, 1, 1),
       (1, 0, 0),(1, 0, 1),(1, 1, 0),(1, 1, 1)],
}
VoxelPointOrder={
    1:[(0,),(1,)],
    # Figure 3.6.1 in Numerical recips
    2:[(0,0),(1,0),(1,1),(0,1)], 
    # 
    3:[(0,0,0),(1,0,0),(1,1,0),(0,1,0),
       (0,0,1),(1,0,1),(1,1,1),(0,1,1),],
}

p=lambda x,k: 0 if k<0 else x**k

X1=np.zeros((4,4))
for k,(x,) in enumerate(VoxelPointOrder[1]):
    X1[k,:]=np.array([p(x,i) for i in range(4)])
    X1[k+2,:]=np.array([i*p(x,i-1) for i in range(4)])
W1=np.linalg.inv(X1)

X2=np.zeros((16,16))
for k,(x,y) in enumerate(VoxelPointOrder[2]):
    X2[k,:]=np.array([p(x,i)*p(y,j) for i,j in product(range(4),range(4))])
    X2[k+4,:]=np.array([i*p(x,i-1)*p(y,j) for i,j in product(range(4),range(4))])
    X2[k+8,:]=np.array([j*p(x,i)*p(y,j-1) for i,j in product(range(4),range(4))])
    X2[k+12,:]=np.array([i*j*p(x,i-1)*p(y,j-1) for i,j in product(range(4),range(4))])
W2=np.linalg.inv(X2) 

X3=np.zeros((64,64))
for n,(x,y,z) in enumerate(VoxelPointOrder[3]):
    X3[n,:]   =np.array([p(x,i)*p(y,j)*p(z,k) for i,j,k in product(range(4),range(4),range(4))])
    X3[n+8,:] =np.array([i*p(x,i-1)*p(y,j)*p(z,k) for i,j,k in product(range(4),range(4),range(4))])
    X3[n+16,:]=np.array([j*p(x,i)*p(y,j-1)*p(z,k) for i,j,k in product(range(4),range(4),range(4))])
    X3[n+24,:]=np.array([k*p(x,i)*p(y,j)*p(z,k-1) for i,j,k in product(range(4),range(4),range(4))])
    X3[n+32,:]=np.array([k*j*p(x,i)*p(y,j-1)*p(z,k-1) for i,j,k in product(range(4),range(4),range(4))])    
    X3[n+40,:]=np.array([i*k*p(x,i-1)*p(y,j)*p(z,k-1) for i,j,k in product(range(4),range(4),range(4))])
    X3[n+48,:]=np.array([i*j*p(x,i-1)*p(y,j-1)*p(z,k) for i,j,k in product(range(4),range(4),range(4))])
    X3[n+56,:]=np.array([i*j*k*p(x,i-1)*p(y,j-1)*p(z,k-1) for i,j,k in product(range(4),range(4),range(4))])
W3=np.linalg.inv(X3)
np.savetxt('W1_4x4.txt',W1.flatten('F').astype(np.int),fmt='%d', delimiter=',',newline='\n')
np.savetxt('W2_16x16.txt',W2.flatten('F').astype(np.int),fmt='%d', delimiter=',',newline='\n')
np.savetxt('W3_64x64.txt',W3.flatten('F').astype(np.int),fmt='%d', delimiter=',',newline='\n')
#W1.flatten(order='F'),W2.flatten(order='F'),W3.flatten(order='F')

def cucof(ys,y1s,ds,W=None):
    """
    ys,y1s : the value of f,f_x in grid points (0,0),(0,1),(1,0),(1,1)
    d1 : the length of the 1th and 2th dimension
    """
    d1=ds[0]
    x=np.concatenate([ys,y1s*d1])
    c=np.dot(W,x).reshape((4,),order='C')
    return c
cucof=partial(cucof,W=W1)
def cuint(c,seeds,xs):
    x=xs[0]
    d1=seeds[0][1]-seeds[0][0]
    t=(x-seeds[0][0])/d1
    y=((c[3]*t+c[2])*t+c[1])*t+c[0]
    y1=(3*c[3]*t+2*c[2])*t+c[1]
    return y,(y1/d1,)

# Press W H, Teukolsky S A, Flannery B P, et al. Numerical recipes in Fortran 77: volume 1, volume 1 of Fortran numerical recipes: the art of scientific computing[M]. Cambridge university press, 1992.
def bcucof(ys,y1s,y2s,y12s,ds,W=None):
    """
    ys,y1s,y2s,y12s : the value of f,f_x,f_y,f_xy in grid points (0,0),(0,1),(1,0),(1,1)
    d1,d2 : the length of the 1th and 2th dimension
    """
    d1,d2=ds
    x=np.concatenate([ys,y1s*d1,y2s*d2,y12s*(d1*d2)])
    c=np.dot(W,x).reshape((4,4),order='C')
    return c
bcucof=partial(bcucof,W=W2)

def bcuint(c,seeds,xs):
    d1,d2=[seed[1]-seed[0] for seed in seeds]
    t,u=[(xs[i]-seeds[i][0])/(seeds[i][1]-seeds[i][0]) for i in range(2)]

    y,y1,y2=0.,0.,0.
    for i in range(3,-1,-1):
        y=t*y+((c[i,3]*u+c[i,2])*u+c[i,1])*u+c[i,0]
        y1=u*y1+(3*c[3,i]*t+2*c[2,i])*t+c[1,i]
        y2=t*y2+(3*c[i,3]*u+2*c[i,2])*u+c[i,1]
    return y,(y1/d1,y2/d2)

def tricucof(ys,y1s,y2s,y3s,y23s,y13s,y12s,y123s,ds,W=None):
    d1,d2,d3=ds
    x=np.concatenate([ys,y1s*d1,y2s*d2,y3s*d3,
                      y23s*(d2*d3),y13s*(d1*d3),y12s*(d1*d2),
                      y123s*(d1*d2*d3)])
    c=np.dot(W,x).reshape((4,4,4),order='C')
    return c
tricucof=partial(tricucof,W=W3)

def tricuint(c,seeds,xs):
    """
    seeds: tuple of lower bound and upper bound of axis
        ((x1l,x1u),(x2l,x2u),(x3l,x3u))
    """
    d1,d2,d3=[seeds[i][1]-seeds[i][0] for i in range(3)]
    r,s,t=[(xs[i]-seeds[i][0])/(seeds[i][1]-seeds[i][0]) for i in range(3)]
    #print(c)
    
    y,y1,y2,y3=0.,0.,0.,0.
    for i in range(3,-1,-1):
        inc,inc1,inc2,inc3=0.,0.,0.,0.
        for j in range(3,-1,-1):
            inc=t*inc+((c[i,3,j]*s+c[i,2,j])*s+c[i,1,j])*s+c[i,0,j]
            inc1=s*inc1+(3*c[3,j,i]*r+2*c[2,j,i])*r+c[1,j,i]
            inc2=t*inc2+(3*c[i,3,j]*s+2*c[i,2,j])*s+c[i,1,j]
            inc3=r*inc3+(3*c[j,i,3]*t+2*c[j,i,2])*t+c[j,i,1]
        y=r*y+inc
        y1=t*y1+inc1
        y2=r*y2+inc2
        y3=s*y3+inc3
    return y,(y1/d1,y2/d2,y3/d3)

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
        
    def isSignChange(self,vindex,X):
        """
        whether if there is sign change in the voxcel
        """
        #考虑浮点数误差
        vals=X[tuple(slice(ind,ind+2) for ind in vindex)].flatten()
        return np.abs(np.mean(np.where(np.abs(vals)>1e-13,
                                       np.sign(vals),0)
                             ))!=1
    def isPoint(self,ptindex):
        """
        返回节点是否是内部节点
        """
        return all(0<=ind<l for l,ind in zip(self.shape,ptindex))

    def coord(self,ptindex):
        return [seed[ind] for seed,ind in zip(self.seeds,ptindex)]

    def neighbours(self,index):
        """
        返回邻接节点的索引列表
        """
        ndim=self.ndim
        return [tuple((index[k]+j*(k==i) for k in range(ndim))) 
                        for i in range(ndim) for j in (-1,1) 
                        if 0<=index[i]+j<self.shape[i]]

    def adjacent(self,ptindex,axis,N=1):
        """
        返回 沿axis方向左右两侧与节点index相邻的N个节点的列表的列表
        
        """
        ndim=self.ndim
        
        pts=[[tuple([ptindex[k]-i*(k==axis) for k in range(ndim)]) for i in range(1,N+1)],
             [tuple([ptindex[k]+i*(k==axis) for k in range(ndim)]) for i in range(1,N+1)]]
        return pts

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

    def voxelPoints(self,vindex):
        """
        return a list of index of the grid points of voxel `vindex`
        """
        ndim=self.ndim
        return [tuple(vindex[i]+pt[i] for i in range(ndim)) 
                    for pt in VoxelPointOrder[ndim] ]
    
    def voxelBound(self,vindex):
        return [(seed[ind],seed[ind+1]) for seed,ind in zip(self.seeds,vindex)]
    
    def voxelValues(self,vindex,X):
        """
        return multi dimension array of values 
        """
        return X[tuple(slice(ind,ind+2) for ind in vindex)]
    
    def triVoxelValues(self,vindex,X):
        return np.array([X[ptindex] for ptindex in self.voxelPoints(vindex)])
    
    def flatten(self,X):
        """
        flatten a multi dimension array `X` in the order defined in `VoxelPointOder`
        """
        return np.array([X[pt] for pt in VoxelPointOrder[self.ndim]])
    
    def cubicCoff(self,vindex,X):
        """
        Using Bicubic or Tricubic Interpolation to Calculate The distance of nodes in those elements which the zero level set crosses 
        1. Chopp D L. Some improvements of the fast marching method[J]. SIAM Journal on Scientific Computing, 2001, 23(1): 230-244.
            Section 3.2
        2. Sukumar N, Chopp D L, Béchet E, et al. Three‐dimensional non‐planar crack growth by a coupled extended finite element and fast marching method[J]. International journal for numerical methods in engineering, 2008, 76(5): 727-748.
            Section 3.1.1 
        """
        grid=self
        ndim=grid.ndim
        oneOrder=lambda dx1,dx2: (-(dx2)/(dx1 * (dx1 + dx2)),(dx2 - dx1) / (dx1 * dx2),dx1 / (dx2 * (dx1 + dx2)))
        
        if ndim==1:
            # ys,y1s
            vals=np.array([X[i] for i in 
                           (max(0,min(j,grid.shape[0]-1)) for j in range(vindex[0]-1,vindex[0]+3))])
            ds=np.array([grid.seeds[0][j+1]-grid.seeds[0][j] for j in 
                         (max(0,min(i,grid.shape[0]-2)) for i in range(vindex[0]-1,vindex[0]+2))])
            ys=vals[1:3]
            y1s=np.zeros((2,))
            for i in range(1,3):
                a,b,c=oneOrder(ds[i-1],ds[i])
                y1s[i-1]=a*vals[i-1]+b*vals[i]+c*vals[i+1]
            return cucof(ys,y1s,[grid.seeds[i][vindex[i]+1]-grid.seeds[i][vindex[i]] for i in range(ndim)])
        elif ndim==2:
            #ys,y1s,y2s,y12s
            vals,ds=np.zeros((4,4)),np.zeros((2,3))
            ys=np.zeros((2,2))
            y1s,y2s=np.zeros((2,2)),np.zeros((2,2))
            y12s=np.zeros((2,2))
            
            for i in range(4):
                ind1=min(max(vindex[0]-1+i,0),grid.shape[0]-1)
                for j in range(4):
                    ind2=min(max(vindex[1]-1+j,0),grid.shape[1]-1)
                    vals[i,j]=X[ind1,ind2]
            for axis in range(2):
                for j in range(3):
                    ind=vindex[axis]-1+j
                    if ind<0 or ind>grid.shape[axis]-2:
                        ds[axis,j]=10000*(grid.seeds[axis][-1]-grid.seeds[axis][0])
                    else:
                        ds[axis,j]=grid.seeds[axis][ind+1]-grid.seeds[axis][ind]
            for i in range(1,3):
                dx1,dx0=ds[0,i],ds[0,i-1]
                for j in range(1,3):
                    dy1,dy0=ds[1,j],ds[1,j-1]
                    ys[i-1,j-1]=vals[i,j]
                    
                    a,b,c=oneOrder(dx0,dx1)
                    y1s[i-1,j-1]=a*vals[i-1,j]+b*vals[i,j]+c*vals[i+1,j]
                    a,b,c=oneOrder(dy0,dy1)
                    y2s[i-1,j-1]=a*vals[i,j-1]+b*vals[i,j]+c*vals[i,j+1]
                    
                    y12s[i-1,j-1]=(vals[i+1,j+1]-vals[i+1,j-1]-vals[i-1,j+1]+vals[i-1,j-1])/((dx0+dx1)*(dy1+dy0))
                    
            return bcucof(self.flatten(ys),self.flatten(y1s),self.flatten(y2s),self.flatten(y12s),
                         [grid.seeds[i][vindex[i]+1]-grid.seeds[i][vindex[i]] for i in range(ndim)])
        elif ndim==3:
            #ys,y1s,y2s,y3s,y12ss,y13s,y23s,y123s
            ys=np.zeros((2,2,2))
            y1s,y2s,y3s=np.zeros((2,2,2)),np.zeros((2,2,2)),np.zeros((2,2,2))
            y23s,y13s,y12s=np.zeros((2,2,2)),np.zeros((2,2,2)),np.zeros((2,2,2))
            y123s=np.zeros((2,2,2))
            
            vals=np.zeros((4,4,4))
            for idx in product(*[range(4) for i in range(ndim)]):
                vals[idx]=X[tuple(min(max(vindex[i]-1+idx[i],0),grid.shape[i]-1) for i in range(ndim))]
            
            ax_dx=[[],]*ndim
            for axis in range(ndim):
                ind=vindex[axis]
                for i in range(-1,2):
                    ind1=ind+i
                    if ind1<0 or ind1>grid.shape[axis]-2:
                        lth=10000*(grid.seeds[axis][-1]-grid.seeds[axis][1])
                    else:
                        lth=grid.seeds[axis][ind1+1]-grid.seeds[axis][ind1]
                    ax_dx[axis].append(lth)
                ax_dx[axis]=np.array(ax_dx[axis])
            
            for idx in product(*[range(2) for i in range(ndim)]):
                i,j,k=tuple(idx[l]+1 for l in range(ndim))
                ys[idx]=vals[i,j,k]
                ds=[(ax_dx[axis][idx[axis]],ax_dx[axis][idx[axis]+1]) for axis in range(ndim)]
                
                a,b,c=oneOrder(*ds[0])
                y1s[idx]=a*vals[i-1,j,k]+b*vals[i,j,k]+c*vals[i+1,j,k]
                a,b,c=oneOrder(*ds[1])
                y2s[idx]=a*vals[i,j-1,k]+b*vals[i,j,k]+c*vals[i,j+1,k]
                a,b,c=oneOrder(*ds[2])
                y3s[idx]=a*vals[i,j,k-1]+b*vals[i,j,k]+c*vals[i,j,k+1]
                
                a=(ds[1][0]+ds[1][1])*(ds[2][0]+ds[2][1])
                y23s[idx]=(vals[i,j+1,k+1]+vals[i,j-1,k-1]-vals[i,j+1,k-1]-vals[i,j-1,k+1])/a
                a=(ds[0][0]+ds[0][1])*(ds[2][0]+ds[2][1])
                y13s[idx]=(vals[i+1,j,k+1]+vals[i-1,j,k-1]-vals[i+1,j,k-1]-vals[i-1,j,k+1])/a
                a=(ds[1][0]+ds[1][1])*(ds[0][0]+ds[0][1])
                y12s[idx]=(vals[i+1,j+1,k]+vals[i-1,j-1,k]-vals[i-1,j+1,k]-vals[i+1,j-1,k])/a   
                
                a=(ds[0][0]+ds[0][1])*(ds[1][0]+ds[1][1])*(ds[2][0]+ds[2][1])
                y123s[idx]=((vals[i+1,j+1,k+1]+vals[i-1,j-1,k+1]-vals[i-1,j+1,k+1]-vals[i+1,j-1,k+1])-
                            (vals[i+1,j+1,k-1]+vals[i-1,j-1,k-1]-vals[i-1,j+1,k-1]-vals[i+1,j-1,k-1]))/a 
            return tricucof(self.flatten(ys),
                    self.flatten(y1s),self.flatten(y2s),self.flatten(y3s),
                    self.flatten(y23s),self.flatten(y13s),self.flatten(y12s),
                    self.flatten(y123s),
                    [grid.seeds[i][vindex[i]+1]-grid.seeds[i][vindex[i]] for i in range(ndim)])
        else:
            raise NotImplementedError('Cubic Interpolation for %d D Not Sumpport Now'%(ndim))

    def cubicInterpolation(self,cof,vindex,xs):
        """
        计算vindex中 xs对应的值和导数
        """
        ndim=self.ndim
        seeds=[(seed[ind],seed[ind+1]) for seed,ind in zip(self.seeds,vindex)]
        #seeds=[(self.seeds[i][vindex[i]],self.seeds[i][vindex[i]+1]) for i in range(ndim)]
        if ndim==1:
            return cuint(cof,seeds,xs)
        elif ndim==2:
            return bcuint(cof,seeds,xs)
        elif ndim==3:
            return tricuint(cof,seeds,xs)
        else:
            raise NotImplementedError('Cubic Interpolation for %d D Not Sumpport Now'%(ndim))        
    
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