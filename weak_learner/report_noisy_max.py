import numpy as np
from diffprivlib.mechanisms import laplace

def report_noisy_argmax(utilities, epsilon, sensitivity):
    utilities = np.array(utilities)
    lap_mech = laplace.Laplace(epsilon=epsilon, sensitivity=sensitivity)
    for i in range(len(utilities)):
        utilities[i] = lap_mech.randomise(utilities[i])
    return np.argmax(np.array(utilities))