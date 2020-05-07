import numpy as np

def closestTwo(xl,x):
    """
    寻找xl中最靠近x的两个点的下标
    """
    i1,d1=None,np.infty
    i2,d2=None,np.infty
    for i,pt in enumerate(xl):
        d=sum(((x[j]-pt[j])**2 for j in range(len(x))))**0.5
        if d<d1:
            i2,d2=i1,d1
            i1,d1=i,d
        elif d<d2:
            i2,d2=i,d
    x1,x2=xl[i1],xl[i2]
    a=sum(((x[j]-x1[j])**2 for j in range(len(x))))
    b=sum(((x[j]-x2[j])**2 for j in range(len(x))))
    c=sum(((x[j]-x1[j])*(x[j]-x2[j]) for j in range(len(x))))
    # (1-alpha)*x1+alpha*x2 is the closet 
    alpha=min(1,max(0,(a-c)/(a+b-2*c)))
    return i1,i2,alpha