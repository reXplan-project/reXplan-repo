import numpy as np
import matplotlib.pyplot as plt

def fragilityCurveGenerator(V, beta, samples):
	'''
	V is the designed wind speed of the transmission line
	beta is the designed value
	'''
	x = np.linspace(0,1.1*beta*V,samples)

	y = np.zeros(samples)
	for idx, j in enumerate(x):
		if j <= V:
			y[idx] = 0
		elif j >= beta*V:
			y[idx] = 1
		else:
			y[idx] = min(np.exp(0.6931*(j-V)/V)-1,1)

	return x, y

x,y = fragilityCurveGenerator(30, 2, 1000)

np.savetxt("fc1.csv", np.transpose(np.stack((x, y), axis=0)), fmt='%.8f', delimiter=",")

print(np.transpose(np.stack((x, y), axis=0)))

plt.plot(x,y)
plt.show()