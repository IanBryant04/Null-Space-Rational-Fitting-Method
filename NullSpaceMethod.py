import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import null_space
from numpy.polynomial.polynomial import polydiv, polyval
from utils import sturm_count_zeros

def data_handler(df, m):
    f = df.iloc[:, 0].to_numpy(dtype=float)
    X = df.iloc[:, 1].to_numpy(dtype=complex)
    N = len(f)

    f_min, f_max = np.min(f), np.max(f)
    f = 2 * (f - f_min) / (f_max - f_min) - 1

    A_complex = np.zeros((N, 2 * m + 2), dtype=complex)
    for k in range(N):
        for i in range(m + 1):
            A_complex[k, i] = -f[k] ** i
            A_complex[k, i + m + 1] = X[k] * (f[k] ** i)

    A = complex_to_real_matrix(A_complex)
    x = null_space(A).T
    if x.size == 0:
        print("Null space empty — cannot solve.")
        return
    print(f"Null space shape: {x.shape}")

    ysv = np.zeros((x.shape[0], N), dtype=complex)
    for j in range(x.shape[0]):
        for i in range(N):
            ysv[j, i] = myfunction_mm(x[j, :], f[i], m)

    myerr = ysv - X
    mynorm = np.max(np.abs(myerr), axis=1)
    I = np.argmin(mynorm)
    final_sol = ysv[I, :]

    freq_GHz = df.iloc[:, 0].to_numpy() / 1e9
    plt.figure(figsize=(10, 8))

    # Magnitude comparison
    plt.subplot(2, 1, 1)
    plt.plot(freq_GHz, 20 * np.log10(np.abs(X)), "ro", markersize=3, label="Measured |S21| (dB)")
    plt.plot(freq_GHz, 20 * np.log10(np.abs(final_sol)), "b-", linewidth=1.5, label="Fitted |S21| (dB)")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"Rational Fit of S21 (m = {m})")
    plt.legend()
    plt.grid(True)

    # Error plot
    plt.subplot(2, 1, 2)
    plt.plot(freq_GHz, np.abs(myerr[I, :]), "k-", linewidth=1)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Absolute Error")
    plt.title("Error Between Fit and Data")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    minE, maxE = np.min(np.abs(myerr[I, :])), np.max(np.abs(myerr[I, :]))
    print(f"Error range: {minE:.3e} {maxE:.3e}")

    return final_sol, np.abs(myerr[I, :]), np.sum(np.abs(myerr[I, :])**2)


def myfunction_mm(x, s, m):
    L = len(x) // 2
    x_complex = x[:L] + 1j * x[L:]
    rexp = np.array([s**j for j in range(m + 1)], dtype=complex)
    top = np.dot(x_complex[: m + 1], rexp)
    bottom = np.dot(x_complex[m + 1 :], rexp)
    if np.abs(bottom) < 1e-12 or np.isnan(bottom) or np.isinf(bottom):
        bottom = 1e-12
    return top / bottom

def complex_to_real_matrix(A_complex):
    A_real = np.block([[np.real(A_complex), -np.imag(A_complex)],[np.imag(A_complex),  np.real(A_complex)],])
    return A_real

