# -*- coding: utf-8 -*-
"""Assignment-1

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sHfwtNCzX64eYI4t_JVyXvKAMoiaLHAc
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from math import sin

#if using termux
import subprocess
import shlex
#end if


def dir_vec(A,B):
  return B-A

def norm_vec(A,B):
  return np.matmul(omat, dir_vec(A,B))

#Generate line points
#def line_gen(A,B):
#  len =10
#  dim = A.shape[0]
#  x_AB = np.zeros((dim,len))
#  lam_1 = np.linspace(0,1,len)
#  for i in range(len):
#    temp1 = A + lam_1[i]*(B-A)
#    x_AB[:,i]= temp1.T
#  return x_AB

#Generate line intercepts
def line_icepts(n,c):
  e1 = np.array([1,0]) 
  e2 = np.array([0,1]) 
  A = c*e1/(n@e1)
  B = c*e2/(n@e2)
  return A,B

#Generate line points
def line_dir_pt(m,A,k1,k2):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(k1,k2,len)
  for i in range(len):
    temp1 = A + lam_1[i]*m
    x_AB[:,i]= temp1.T
  return x_AB
#Generate line points

def line_norm_eq(n,c,k):
  len =10
  dim = n.shape[0]
  m = omat@n
  m = m/np.linalg.norm(m)
#  x_AB = np.zeros((dim,2*len))
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(k[0],k[1],len)
#  print(lam_1)
#  lam_2 = np.linspace(0,k2,len)
  if c==0:
    for i in range(len):
      temp1 = lam_1[i]*m
      x_AB[:,i]= temp1.T
  else:
    A,B = line_icepts(n,c)
    for i in range(len):
      temp1 = A + lam_1[i]*m
      x_AB[:,i]= temp1.T
#    temp2 = B + lam_2[i]*m
#    x_AB[:,i+len]= temp2.T
  return x_AB

#def line_dir_pt(m,A, dim):
#  len = 10
#  dim = A.shape[0]
#  x_AB = np.zeros((dim,len))
#  lam_1 = np.linspace(0,10,len)
#  for i in range(len):
#    temp1 = A + lam_1[i]*m
#    x_AB[:,i]= temp1.T
#  return x_AB


#Generate line points
def line_gen(A,B):
  len =10
  x_AB = np.zeros((2,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB

#Foot of the Altitude
def alt_foot(A,B,C):
  m = B-C
  n = np.matmul(omat,m) 
  N=np.vstack((m,n))
  p = np.zeros(2)
  p[0] = m@A 
  p[1] = n@B
  #Intersection
  P=np.linalg.inv(N.T)@p
  return P

#Intersection of two lines
def line_intersect(n1,c1,n2,c2):
  N=np.vstack((n1,n2))
  p = np.array([c1,c2]) 
  #Intersection
  P=np.linalg.inv(N)@p
#  P=np.linalg.inv(N.T)@p
  return P

#Code by R.Yamini
#June 1, 2021
#To construct triangle ABC

#Given
angA = 70
print(angA)
angB = 50
print(angB)
b = 3
print(b)

#To find angle C
angC = 180 - (angA + angB) 
print(angC)
#To find sides
#Use formula : a/sinA = b/sinB = c/sinC
#To find a
a= (math.sin(math.radians(angA))/math.sin(math.radians(angB)))*b
print(a)
#To find c
c = (math.sin(math.radians(angC))/math.sin(math.radians(angB)))*b
print(c)

#To find coordinate of B(p,q)
p = (math.cos(math.radians(70)))*c
q = (math.sin(math.radians(70)))*c
print(p,q)

#Triangle vertices
A = np.array([0,0])
print(A)
B = np.array([p,q])
print(B)
C = np.array([b,0])
print(C)

#Generating lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')

plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1 + 0.1), A[1] * (1 - 0.1) , 'A')
plt.plot(B[0], B[1], 'o')
plt.text(B[0] * (1 - 0.2), B[1] * (1) , 'B')
plt.plot(C[0], C[1], 'o')
plt.text(C[0] * (1 + 0.03), C[1] * (1 - 0.1) , 'C')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor