import numpy as np
import cmath
import matplotlib.pyplot as plt
import scipy.io as sio

""" Load System Data """
sys_data = sio.loadmat('system_q1a.mat',squeeze_me=True) # event. use squeeze_me=True to get rid of unnecessary nesting
A = sys_data['A_q1a']
n = len(A)


#%%
eigenvalues, Phi = np.linalg.eig(A)

eigenvalues = np.array(eigenvalues).reshape(n, 1)
Psi = np.linalg.inv(Phi)
# Lambda = np.eye(n)*[eigenvalues]
imaginary_part = eigenvalues.imag
real_part = eigenvalues.real

freq = imaginary_part/(2*np.pi)
damping = -real_part/abs(eigenvalues)