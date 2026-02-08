import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.plots import plot_convergence


# ----------------------------------------------------
# Load trained BER model (log10(BER))
# ----------------------------------------------------
ber_model = joblib.load("/Users/kavanakiran/Documents/AI_Projects/optical-photonics-optimizer/models/ber_model_log.pkl")


# ----------------------------------------------------
# One-hot encoding for modulation format
# ----------------------------------------------------
def encode_modulation(mod):
    return {
        "mod_format_NRZ": int(mod == "NRZ"),
        "mod_format_PAM4": int(mod == "PAM4"),
        "mod_format_QAM16": int(mod == "QAM16"),
    }


# ----------------------------------------------------
# Objective function to MINIMIZE
# Input: wavelength, power, modulation
# Output: log10(BER)
# ----------------------------------------------------
def objective_function(params):
    wavelength_nm, launch_power_dBm, mod_format = params

    # Fixed parameters (can be optimized later)
    temperature_C = 40

    mod_encoded = encode_modulation(mod_format)

    X = pd.DataFrame([{
        "wavelength_nm": wavelength_nm,
        "launch_power_dBm": launch_power_dBm,
        "temperature_C": temperature_C,
        **mod_encoded
    }])

    log_ber = ber_model.predict(X)[0]
    if launch_power_dBm > 4:
        log_ber += 0.6 * (launch_power_dBm - 4)**2

    log_ber += np.random.normal(0, 0.2)
    return log_ber



# ----------------------------------------------------
# Search space definition
# ----------------------------------------------------
search_space = [
    Real(1530.0, 1565.0, name="wavelength_nm"),
    Real(-10.0, 10.0, name="launch_power_dBm"),
    Categorical(["NRZ", "PAM4", "QAM16"], name="mod_format"),
]


# ----------------------------------------------------
# Run Bayesian Optimization
# ----------------------------------------------------
print("\nRunning Bayesian Optimization to minimize BER...\n")

result = gp_minimize(
    func=objective_function,
    dimensions=search_space,
    n_calls=40,        # number of optimization iterations
    n_initial_points=10,
    random_state=42
)


# ----------------------------------------------------
# Print optimal results
# ----------------------------------------------------
best_wavelength = result.x[0]
best_power = result.x[1]
best_modulation = result.x[2]
best_log_ber = result.fun

print("Optimal Parameters Found:")
print(f"  Wavelength       : {best_wavelength:.2f} nm")
print(f"  Launch Power     : {best_power:.2f} dBm")
print(f"  Modulation       : {best_modulation}")
print(f"  Min log10(BER)   : {best_log_ber:.3f}")
print(f"  Min BER (linear) : {10 ** best_log_ber:.3e}")


# ----------------------------------------------------
# Plot convergence
# ----------------------------------------------------
plot_convergence(result)
plt.title("Bayesian Optimization Convergence (BER Minimization)")
plt.tight_layout()
plt.show()
plt.savefig('figures/optimizer_result.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(6,4))
plt.plot(result.func_vals, marker='o')
plt.xlabel("Iteration")
plt.ylabel("log10(BER)")
plt.title("Raw Objective Values per Iteration")
plt.grid(True)
plt.show()