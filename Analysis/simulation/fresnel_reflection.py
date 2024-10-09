
# -*- coding: utf-8 -*-
"""
Examples of plots and calculations using the tmm package.
"""

from __future__ import division, print_function, absolute_import

import tmm

from numpy import pi, linspace, inf, array
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# "5 * degree" is 5 degrees expressed in radians
# "1.2 / degree" is 1.2 radians expressed in degrees
degree = pi/180


"""
Here's a thin non-absorbing layer (corresponding to the Stern layer), on top of glass, plotted for
different layer indices angles.
"""

# list of refractive indices
n_list = [1.51, 1.39, 1.332]
# list of wavenumbers to plot in nm^-1
ds = linspace(0, 20, num=20)
# initialize lists of y-values to plot
Rnorm = []
for d in ds:
    d_list = [inf, d, inf]
    Rnorm.append(tmm.coh_tmm('s', n_list, d_list, 0, 630)['R'])

Rs = array(Rnorm)
print(Rs[0])
plt.figure()
plt.plot(ds, (Rs - Rs[0])/Rs[0], '.-')
plt.xlabel('Layer thickness (nm)')
plt.ylabel('Fraction reflected')
plt.title('Reflection of unpolarized light at normal incidence')
plt.show()

