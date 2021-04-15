This program is for interpolating the band structure from the output of wannier90, specifically 'wannier90_hr.dat'.

Please refer to example.py as an example for usage, the variable 'nkx', 'nky', 'nkz' determine the interpolation resolution; 'k_vec' is the interpolated k-mesh; 'eig' is the interpolated band structure.

Usage:

'python -i example.py'

If you want to plot the fermi surface, please use the 'PyFermi':

https://github.com/lyglst/pyFermi
