import numpy as np

# Basiswerte (cm-System)
rho_base = 2.7e-3   # kg/cm³
E_base = 6.9e6      # N/cm²
nu = 0.33           # (einheitenlos)

print(f"{'rho (kg/cm³)':>15} | {'E (N/cm²)':>15} | {'sqrt(E/rho) (sqrt(N·cm/kg))':>30}")
print("-"*65)

for n in range(-3, 4):       # rho-Faktor 10^n
    for m in range(-3, 4):   # E-Faktor 10^m
        rho_test = rho_base * 10**n
        E_test = E_base * 10**m
        ratio = E_test / rho_test
        freq_estimate = np.sqrt(ratio)  # Maß für Eigenfrequenz-Skala
        print(f"{rho_test:15.3e} | {E_test:15.3e} | {freq_estimate:30.3e}")

