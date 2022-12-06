import numpy as np
import matplotlib.pyplot as plt

# plot the sinc function
# x = np.linspace(-8, 8, 100)
# y = np.sinc(x)
# plt.plot(x,y)
# plt.show()

# plot the supposed expression for cosc (cosx-1)/x function
x2 = np.linspace(0.1, 4, 100)
yc1 = (np.cos(x2)-1)/x2
yc2 = -np.sin(x2/2.0) * np.sinc(x2/np.pi/2) # in numpy numpy.sinc(x) = sinc(pi*x) actually
plt.plot(x2,yc1)
plt.plot(x2,yc2)
plt.show()
