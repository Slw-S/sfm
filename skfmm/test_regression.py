import numpy as np
import skfmm
import time

# Create a simple test array
phi = np.arange(25, dtype=np.float64).reshape(5, 5)
phi[2, 2] = -1  # Set a zero level set

# Create a non-contiguous array (transpose makes it Fortran-ordered)
phi_noncontig = phi.T

# dx, flag, speed as required by fmm.cpp (defaults are usually fine)
dx = [1.0, 1.0]

# Try running skfmm.distance on the non-contiguous array
try:
    result = skfmm.distance(phi_noncontig, dx=dx)
    print("Success! Result shape:", result.shape)
except Exception as e:
    print("Regression detected! Exception:", e)

print("Starting benchmark")

# Use a large array
shape = (4000, 4000)
phi_c = np.ones(shape, order='C')
phi_f = np.ones(shape, order='F')
phi_c[2000, 2000] = -1
phi_f[2000, 2000] = -1

N = 10  # number of repetitions, increase if you want stronger effect

# C-contiguous timing
start = time.time()
for _ in range(N):
    d_c = skfmm.distance(phi_c)
print("C-contiguous time: {:.3f}s".format(time.time() - start))

# F-contiguous timing
start = time.time()
for _ in range(N):
    d_f = skfmm.distance(phi_f)
print("F-contiguous time: {:.3f}s".format(time.time() - start))

# Results check (should still be identical)
print("Results identical:", np.allclose(d_c, d_f))
