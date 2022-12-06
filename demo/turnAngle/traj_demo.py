import numpy as np
import matplotlib.pyplot as plt

#      car trajectory demo
# ---------------------------------

# define sinc and cosc functions
def sinc(x):
    return np.sinc(x/np.pi)
def cosc(x):
    return np.sin(x/2) * sinc(x/2)

# car settings
L = 3 # meters
V = 1 # m/s

t = np.linspace(0,12,100)
xc = 0.0
yc = 0.0

# choose periodic turn angle setting
phivec = np.linspace(-0.4, 0.4,5)
colors = "kbcgr"
for i,phi in enumerate(phivec):
    arg = V*phi*t/L
    x = V*t*cosc(arg)+xc
    y = V*t*sinc(arg)+yc

    # plot the path
    plt.plot(x,y,f"{colors[i]}-",linewidth=2,label=f"phi={round(phi,1)}")
plt.legend()
plt.title("Turn Angle Space")
plt.show()