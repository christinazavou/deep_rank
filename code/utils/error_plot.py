import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.1, 4, 0.5)
y = np.exp(-x)
yerr = 0.1 + 0.2*np.sqrt(x)
plt.errorbar(x, y, yerr=yerr, fmt='o')
plt.title('Vert. symmetric')
plt.show()
