import h5py, numpy as np, matplotlib.pyplot as plt
f = h5py.File("data/csi_quick.h5","r")
k = list(f.keys())[0]  # first position group
csi = f[k]['csi_re'][:] + 1j*f[k]['csi_im'][:]
H0 = csi[0,:,0,0]
H1 = csi[1,:,0,0]

plt.figure(); plt.plot(np.abs(H0), label='snap0'); plt.plot(np.abs(H1), label='snap1')
plt.title("CSI magnitude"); plt.legend(); plt.show()

plt.figure(); plt.plot(np.unwrap(np.angle(H0)), label='snap0'); plt.plot(np.unwrap(np.angle(H1)), label='snap1')
plt.title("CSI phase (unwrap)"); plt.legend(); plt.show()
