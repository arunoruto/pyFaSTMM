"""Two-sphere cluster: Mueller matrix, cross sections, and T-matrix."""

from pyfastmm import FaSTMM2

f = FaSTMM2()

coords = [[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]]
radii = [1.0, 1.0]
eps = [3.0 + 0.1j, 3.0 + 0.1j]  # permittivity = (refractive index)**2
k = 1.2  # wavenumber

result = f.solve(coords, radii, eps, k, N_theta=91, N_phi=16)
print(f"Cext = {result['c_ext']:.6f}")
print(f"Cabs = {result['c_abs']:.6f}")
print(f"Csca = {result['c_sca']:.6f}")
print(f"Asymmetry parameter = {result['asymmetry']:.6f}")
print(f"Mueller matrix shape: {result['mueller'].shape}")

# Orientation-averaged (randomly oriented cluster)
avg = f.solve(coords, radii, eps, k, N_theta=91, N_phi=16, N_ave=50)
print(f"\nOrientation-averaged Cext = {avg['c_ext']:.6f}")

# T-matrix of the cluster
tmat = f.compute_tmatrix(coords, radii, eps, k, t_order=12)
print(f"\nT-matrix Taa shape: {tmat['Taa'].shape}")
