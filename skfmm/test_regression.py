import numpy as np
import skfmm

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
