# -*- coding: UTF-8 -*-
import numpy as np
from itertools import product
from functools import partial

from constants import VoxelPointOrder

__all__=[
    'cucof','cuint','W1','bcucof','bcuint','W2',
    'tricucof','tricuint','W3','CubicInterpolate'
]
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

## used in Fortran So flatten as 'F'
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

class CubicInterpolate:
    def __init__(self,grid,X):
        self.grid=grid
        self.X=X

        self.coffs=dict()

    def flatten(self,X):
        """
        flatten a multi dimension array `X` in the order defined in `VoxelPointOder`
        """
        return np.array([X[pt] for pt in VoxelPointOrder[self.grid.ndim]])

    def cubicCoff(self,vindex):
        """
        Using Bicubic or Tricubic Interpolation to Calculate 
        the distance of nodes in those elements which the zero level set crosses 
        
        Reference
        _______
        
        1. Chopp D L. Some improvements of the fast marching method[J]. SIAM Journal on Scientific Computing, 2001, 23(1): 230-244.
            Section 3.2
        2. Sukumar N, Chopp D L, Béchet E, et al. Three‐dimensional non‐planar crack growth by a coupled extended finite element and fast marching method[J]. International journal for numerical methods in engineering, 2008, 76(5): 727-748.
            Section 3.1.1 
        """ 
        grid=self.grid
        ndim=grid.ndim
        X=self.X
        
        if vindex in self.coffs:
            return self.coffs[vindex]

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
            
            cof=cucof(ys,y1s,[grid.seeds[i][vindex[i]+1]-grid.seeds[i][vindex[i]] for i in range(ndim)])
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
                    
            cof=bcucof(self.flatten(ys),self.flatten(y1s),self.flatten(y2s),self.flatten(y12s),
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
            cof=tricucof(self.flatten(ys),
                    self.flatten(y1s),self.flatten(y2s),self.flatten(y3s),
                    self.flatten(y23s),self.flatten(y13s),self.flatten(y12s),
                    self.flatten(y123s),
                    [grid.seeds[i][vindex[i]+1]-grid.seeds[i][vindex[i]] for i in range(ndim)])
        else:
            raise NotImplementedError('Cubic Interpolation for %d D Not Sumpport Now'%(ndim))
        
        self.coffs[vindex]=cof
        return cof

    def interpolate(self,xs):
        """
        计算vindex中 xs对应的值和导数
        """
        vindex=self.grid.locate(xs)

        if vindex in self.coffs:
            cof=self.coffs[vindex]
        else:
            cof=self.cubicCoff(vindex)
        
        ndim=self.grid.ndim
        seeds=[(seed[ind],seed[ind+1]) for seed,ind in zip(self.grid.seeds,vindex)]
        #seeds=[(self.seeds[i][vindex[i]],self.seeds[i][vindex[i]+1]) for i in range(ndim)]
        if ndim==1:
            return cuint(cof,seeds,xs)
        elif ndim==2:
            return bcuint(cof,seeds,xs)
        elif ndim==3:
            return tricuint(cof,seeds,xs)
        else:
            raise NotImplementedError('Cubic Interpolation for %d D Not Sumpport Now'%(ndim))
