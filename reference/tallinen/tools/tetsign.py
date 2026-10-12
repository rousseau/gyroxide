import sys, numpy as np
def read(p):
    f=open(p); nn=int(f.readline()); X=np.loadtxt(f,max_rows=nn)
    ne=int(f.readline()); T=np.loadtxt(f,max_rows=ne,dtype=np.int64)[:,1:]-1
    return X,T[:,[0,1,3,2]]
for p in sys.argv[1:]:
    X,T=read(p); a=X[T[:,1]]-X[T[:,0]]; b=X[T[:,2]]-X[T[:,0]]; c=X[T[:,3]]-X[T[:,0]]
    v=np.einsum('ij,ij->i',a,np.cross(b,c))/6
    print(p.split('/')[-1], 'neg',(v<0).sum(),'pos',(v>0).sum(),'bbox',X.min(0).round(3),X.max(0).round(3))
