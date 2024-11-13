
import numpy as np
import cmath
import matplotlib.pyplot as plt
import scipy.io as sio


""" Load System Data """

sys_data = sio.loadmat('system_q1a.mat',squeeze_me=True) # event. use squeeze_me=True to get rid of unnecessary nesting
A_a = sys_data['A_q1a']

sys_data = sio.loadmat('system_q1b.mat',squeeze_me=True) # event. use squeeze_me=True to get rid of unnecessary nesting
A_b = sys_data['A_q1b']

sys_data = sio.loadmat('system_q1c.mat',squeeze_me=True) # event. use squeeze_me=True to get rid of unnecessary nesting
A_c = sys_data['A_q1c']

def eigenproperty(A):
    n = len(A)
    eigenvalues, Phi = np.linalg.eig(A)
    eigenvalues = np.array(eigenvalues)
    Psi = np.linalg.inv(Phi)
    # Lambda = np.eye(n)*[eigenvalues]
    imaginary_part = eigenvalues.imag
    real_part = eigenvalues.real

    freq = imaginary_part/(2*np.pi)
    damping = -real_part/abs(eigenvalues)
    return eigenvalues, freq, damping, Phi, Psi, n

eigenvalues_a, freq_a, damping_a, Phi_a, Psi_a, n_a = eigenproperty(A_a)
eigenvalues_b, freq_b, damping_b, Phi_b, Psi_b, n_b = eigenproperty(A_b)
eigenvalues_c, freq_c, damping_c, Phi_c, Psi_c, n_c = eigenproperty(A_c)

x0_a = np.zeros([n_a])
x0_a[0] = 5*np.pi/180 #radians 5°

x0_b = np.zeros([n_b])
x0_b[0] = 5*np.pi/180 #radians 5°

x0_c = np.zeros([n_c])
x0_c[0] = 5*np.pi/180 #radians 5°
#%%
t_end = 5  #time after which simulation stpped
t = np.linspace(0, t_end, 1000)

delta_x_a =np.zeros((len(t),n_a))
delta_x_b =np.zeros((len(t),n_b))
delta_x_c =np.zeros((len(t),n_c))

for i in range(len(t)):

    # Calculate delta_x at time t
    delta_x_a[i,:] = Phi_a @ (np.exp(eigenvalues_a*t[i]) * (Psi_a @ x0_a))
    delta_x_b[i,:] = Phi_b @ (np.exp(eigenvalues_b*t[i]) * (Psi_b @ x0_b))
    delta_x_c[i,:] = Phi_c @ (np.exp(eigenvalues_c*t[i]) * (Psi_c @ x0_c))
    
plt.figure(figsize=(10, 6))

plt.plot(t, delta_x_a[:, 0]*180/np.pi, label="$\Delta \delta$ manual excitation ", color='b')
plt.plot(t, delta_x_b[:, 0]*180/np.pi, label="$\Delta \delta$ with AVR", color='r')
plt.plot(t, delta_x_c[:, 0]*180/np.pi, label="$\Delta \delta$ with AVR and PSS",color='g')
plt.xlabel('Time [s]')
plt.ylabel("Rotor angle deviation $\Delta \delta$ [°]")
plt.legend(loc='lower left', ncol=1)
plt.xlim(0, t_end)
plt.grid()
plt.title('Rotor angle deviation with different excitations')
plt.savefig('Task_1_3.png', dpi=300, bbox_inches='tight')
plt.show
