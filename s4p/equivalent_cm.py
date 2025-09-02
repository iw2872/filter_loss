#!/usr/bin/env python3
"""
equivalent_4p.py

 - Creates a 4-port EMI filter model from RLC parameters.
 - This script takes the RLC values for the CM choke (R_cm, L_cm, C_cm) and
 - a differential mode capacitor (C_dm) and calculates the full 4-port S-parameters.
"""

import numpy as np
import re
from math import sqrt
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def cm_choke_model_impedance(freqs, R, L, C):
    """
    Calculates the impedance of a CM choke equivalent circuit.
    Model: A series R-L branch in parallel with a parasitic capacitance C.
    """
    omega = 2 * np.pi * freqs
    Z_series = R + 1j * omega * L
    Z_parallel = 1 / (1j * omega * C)

    # Total impedance of the choke in a parallel configuration
    Z_choke = (Z_series * Z_parallel) / (Z_series + Z_parallel)
    return Z_choke


def emi_filter_model_s4p(freqs, R_cm, L_cm, C_cm, C_dm, R_dm):
    """
    Calculates the 4-port S-parameters for an EMI filter.
    The model includes a CM choke (R_cm, L_cm, C_cm) and differential capacitors (C_dm) with ESR.
    """
    z0 = 50.0  # System impedance

    # Calculate impedances for the CM choke and DM capacitors
    Z_cm = cm_choke_model_impedance(freqs, R_cm, L_cm, C_cm)
    omega = 2 * np.pi * freqs
    Z_dm = R_dm + 1 / (1j * omega * C_dm)

    # Convert impedances to Y-parameters (admittance)
    Y_cm = 1 / Z_cm
    Y_dm = 1 / Z_dm

    # Calculate the Y-parameters for the full 4-port network
    # Y-parameter matrix Y_4x4
    #  [ Y11 Y12 Y13 Y14 ]
    #  [ Y21 Y22 Y23 Y24 ]
    #  [ Y31 Y32 Y33 Y34 ]
    #  [ Y41 Y42 Y43 Y44 ]
    Y_4x4 = np.zeros((4, 4, len(freqs)), dtype=complex)

    Y_4x4[0, 0, :] = Y_dm + Y_cm
    Y_4x4[1, 1, :] = Y_dm + Y_cm
    Y_4x4[2, 2, :] = Y_dm + Y_cm
    Y_4x4[3, 3, :] = Y_dm + Y_cm

    Y_4x4[0, 1, :] = -Y_dm
    Y_4x4[1, 0, :] = -Y_dm
    Y_4x4[2, 3, :] = -Y_dm
    Y_4x4[3, 2, :] = -Y_dm

    Y_4x4[0, 2, :] = -Y_cm
    Y_4x4[2, 0, :] = -Y_cm
    Y_4x4[1, 3, :] = -Y_cm
    Y_4x4[3, 1, :] = -Y_cm

    # Convert Y-parameters to Z-parameters (Z = inv(Y))
    Z_4x4 = np.zeros_like(Y_4x4)
    for i in range(len(freqs)):
        Z_4x4[:, :, i] = np.linalg.inv(Y_4x4[:, :, i])

    # Convert Z-parameters to S-parameters
    S_4x4 = np.zeros_like(Y_4x4)
    for i in range(len(freqs)):
        I = np.eye(4)
        Z_term = Z_4x4[:, :, i] + z0 * I
        Z_inv = np.linalg.inv(Z_term)
        S_4x4[:, :, i] = (Z_4x4[:, :, i] - z0 * I).dot(Z_inv)

    return S_4x4


def main():
    # 🔹 Set the RLC values from your previous fitting result here
    R_cm = 1.0e+01  # Ohms
    L_cm = 1.0e-04  # Henries (100 uH)
    C_cm = 1.0e-12  # Farads (1 pF)

    # 🔹 Define the differential mode capacitor value here
    C_dm = 1.0e-09  # Farads (1 nF)
    R_dm = 1.0e-01  # Ohms (ESR of C_dm)

    # Create a frequency range for simulation
    freqs = np.logspace(1, 8, 200)

    print("🔍 Calculating 4-port S-parameters...")
    S_4x4 = emi_filter_model_s4p(freqs, R_cm, L_cm, C_cm, C_dm, R_dm)

    # Extract differential-mode (Sdd21) and common-mode (Scc21) insertion loss
    # Sdd21 = (S41-S42-S31+S32)/2
    Sdd21 = (S_4x4[3, 0, :] - S_4x4[3, 1, :] - S_4x4[2, 0, :] + S_4x4[2, 1, :]) / 2

    # Scc21 = (S41+S42+S31+S32)/2
    Scc21 = (S_4x4[3, 0, :] + S_4x4[3, 1, :] + S_4x4[2, 0, :] + S_4x4[2, 1, :]) / 2

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, 20 * np.log10(np.abs(Sdd21)), label='Sdd21 (Differential Mode)')
    plt.plot(freqs, 20 * np.log10(np.abs(Scc21)), label='Scc21 (Common Mode)')

    plt.title('Simulated EMI Filter Performance')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Insertion Loss (dB)')
    plt.xscale('log')
    plt.grid(True, which="both")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
