
from numpy import linalg as LAn
import sys
from math import sqrt
from cvxopt import matrix
from cvxopt.blas import dot 
from cvxopt.solvers import qp, options 
import matplotlib.pylab as plt
import numpy as np
import random
#from scipy.linalg import inv,sqrtm,eig
import scipy.linalg as LA
import pandas as pd



def compute_nk(Xk=None,tau=None):
	l=[]
	for c in range(Xk.shape[1]):
		nkc=[]
		mycol = Xk[:,c]
		#print 'col=',mycol
		for i,p in enumerate(mycol):
			if i<1:
                            continue
                        else:
			    nkc.append(p-mycol[i-1]*1.0/tau)
		l.append(nkc)
	nk = np.array(l)
	#print 'Xk\n',Xk
	#print 'nk\n', nk
	return nk

def compute_Ex(X0=None,Xk=None,Gs=None,Htilde=None,Ga=None,tau=None):
	nk = compute_nk(Xk=Xk,tau=tau)
	vk = nk*1.0/tau
	XGsX = 0.5*(X0.transpose().dot(Gs)).dot(X0) 
	vHtildev = tau*1.0*( ((vk.transpose()).dot(Htilde)).dot(vk)).trace()
	xGav     = tau*1.0*( ((Xk).dot(Ga)).dot(vk)).trace()
        #neglecting epsilon.transpose().dot(X0) contribution (fixed-cost)
	Ex=XGsX+vHtildev+xGav
	return Ex

def compute_Vx(Xk=None,C=None,tau=None):
        Xkl=Xk[1:,:]
	Vx =  tau*1.0*(((Xkl).dot(C)).dot(Xkl.transpose())).trace()
	return Vx

def get_L(A=None,lam=None):
	lA = lam*1.0*A
	w, v = LA.eig(lA)
	D= np.diag(w)
	#print D
	#print 'eigs=\t',w
	#print 'eigvecs=\t',v
	return D,v

def get_Xk(D=None,v=None,N=None,z0=None,sqrt_inv_Htilde=None):
	l=[]
	for tk in range(N):
		Dp = LAn.matrix_power(D,tk)
                #print 'tk=',tk
                #print 'D=',D
                #print 'Dp= \n',Dp
                Dp = (v.transpose()).dot(Dp).dot(v)
		zk = Dp.dot(z0)
		xk = (sqrt_inv_Htilde.dot(v)).dot(zk)
		l.append(xk.real)
	Xk = np.array(l)
	return Xk


def get_min_impact_min_var(X0=None,N=None):
    return [ X0*(N-n)*1/N for n in range(N+1)] 





n=2             #no. securities
gamma= 0.5      #2.5*10**-2    #linear permanent impact [ ($/share)/share ] gamma g(v_k) = gamma*v ; E<-sum_k( t*x_k*g(v))
eta  = 2.5      #2.5*10**2     #Hij = Hij*deltaij*eta_i [ ($/share)*1/(share/time)] ; h(v_k) = eta n_k/t ; E<-sum_k(   n_k*h(v))
x0   = 2500     #1.0*10**3     #amt to trade (buy/sell)
N = 4           #no. trade intervals
T = N           #no. time intervals
tau = T*1.0/N   #time per trade interval


#vector of initial holdings
X0 = np.array([ np.rint( random.uniform(0,1)*random.choice([-1,1])*x0)  for x in range(n)])
#X0 = np.array([ np.rint(x0)  for x in range(n)])

# Hij = deltaij*eta_ij
H =  np.diag(np.array([eta  for x in range(n)]))
# Gij = deltaij*gamma_ij
G =  np.diag(np.array([gamma for x in range(n)]))

b =  np.random.uniform(0,1, size=(n,n))
C =  0.5*(b + b.transpose())
#C *= 0.01
Hs = 0.5*(H+H.transpose())
Gs = 0.5*(G+G.transpose())
Ga = 0.5*(G-G.transpose())


Htilde = Hs - 0.5*tau*Gs    #tau*Gs (subtract time*$/share*share/time )
inv_Htilde = LA.inv(Htilde)
sqrt_Htilde = LA.sqrtm(Htilde)
sqrt_inv_Htilde =  LA.sqrtm(inv_Htilde)


A = (sqrt_inv_Htilde.dot(C)).dot(sqrt_inv_Htilde.transpose())
B = (sqrt_inv_Htilde.dot(Ga)).dot(sqrt_inv_Htilde.transpose())


evl=[]
Xl=[]

laml = np.logspace(-6,0)
#lamnl=-1*np.logspace(-6,-0)
#laml =np.sort( list(laml)+list(lamnl))

print laml

for lam in laml:
	D,v = get_L(A=A,lam=lam)                                        #diagonalize lambda*A
	z0 = (v.transpose().dot(sqrt_Htilde)).dot(X0)                   #z0 = vT.(y0) = vT.sqrt_Htilde.X0
	Xk = get_Xk(D=D,v=v,N=N,z0=z0,sqrt_inv_Htilde=sqrt_inv_Htilde)  #Xk = sqrt_inv_Htilde.v.zk; zk = A**k.z0
	Ex = compute_Ex(X0=X0,Xk=Xk,Gs=Gs,Htilde=Htilde,Ga=Ga,tau=tau)
	Vx = compute_Vx(Xk=Xk,C=C,tau=tau)
	Ux = Ex+lam*Vx
	evl.append([Ex,Vx,Ux,lam])
	Xl.append(Xk)
        print 'lam={};Ex={};Vx={}'.format(lam,Ex,Vx)
pe=  [x[0] for x in evl]
pv=  [x[1] for x in evl]
ul=  [x[2] for x in evl]
laml=[x[3] for x in evl]
s= pd.DataFrame()
s['E'] = pe
s['V'] = pv
s['U'] = ul
s['L'] = laml
s['X'] = Xl

print s



minEx = s.loc[s.E==s.E.min()].X.values.tolist()[0]
minVx = s.loc[s.V==s.V.min()].X.values.tolist()[0]
minE =  s.loc[s.E==s.E.min()][['V','E']].values.tolist()[0]
minV =  s.loc[s.V==s.V.min()][['V','E']].values.tolist()[0]
minIm = get_min_impact_min_var(X0=X0,N=N)
plt.plot(pv,pe)
plt.scatter(minV[0],minV[1])
plt.scatter(minE[0],minE[1])
plt.show()

plt.plot(minEx,label='minExp(C)',marker='v')
plt.plot(minVx,label='minVar(C)',linestyle='--',marker='o')
plt.plot(minIm,label='minImp(C)',linestyle=':',marker='.')
plt.legend()
plt.show()
