import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
from datetime import datetime
import logging

from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.plots import plot_convergence

from channel_model import simulate_channel, snr_dB_to_ber

# ============================================================
# SETUP LOGGING AND OUTPUT DIRECTORIES
# ============================================================


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create directories with absolute paths
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Create directories
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Setup logging
log_filename = os.path.join(LOGS_DIR, f'pipeline_{TIMESTAMP}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # Also print to console
    ]
)

logger = logging.getLogger(__name__)

logger.info("="*60)
logger.info("OPTICAL PHOTONICS OPTIMIZATION PIPELINE")
logger.info(f"Run timestamp: {TIMESTAMP}")
logger.info("="*60)

# ============================================================
# LOADING MODELS
# ============================================================

logger.info("Loading trained models...")
try:
    with open("/Users/kavanakiran/Documents/AI_Projects/optical-photonics-optimizer/models/snr_model.pkl", "rb") as f:
        snr_model = pickle.load(f)
    
    with open("/Users/kavanakiran/Documents/AI_Projects/optical-photonics-optimizer/models/ber_model_log.pkl", "rb") as f:
        ber_model = pickle.load(f)
    
    logger.info("Models loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    raise

def encode_modulation(mod_format):
    """
    Convert modulation format string to one-hot encoding
    """
    return {
        "mod_format_NRZ": int(mod_format == "NRZ"),
        "mod_format_PAM4": int(mod_format == "PAM4"),
        "mod_format_QAM16": int(mod_format == "QAM16"),
    }


def predict_performance(wavelength_nm, launch_power_dBm, temperature_C, mod_format):
    """
    Use ML model to predict SNR, then physics formula to calculate BER
    Note: SNR model already learned modulation penalties (they're in training data)
    Returns: (predicted_snr, predicted_ber)
    """
    # Encode the modulation format
    mod_encoded = encode_modulation(mod_format)
    
    # Create feature DataFrame (same format as training data)
    X = pd.DataFrame([{
        "wavelength_nm": wavelength_nm,
        "launch_power_dBm": launch_power_dBm,
        "temperature_C": temperature_C,
        **mod_encoded
    }])
    
    # Predict SNR using ML model (already includes modulation penalty!)
    predicted_snr = snr_model.predict(X)[0]
    
    # Calculate BER directly from predicted SNR (no additional penalty needed)
    predicted_ber = snr_dB_to_ber(predicted_snr)
    
    return predicted_snr, predicted_ber

# ============================================================
# TEST PREDICTION
# ============================================================

logger.info("\nTesting prediction function...")
test_snr, test_ber = predict_performance(
    wavelength_nm=1550,
    launch_power_dBm=0,
    temperature_C=25,
    mod_format="NRZ"
)
logger.info(f"Test prediction: SNR={test_snr:.2f} dB, BER={test_ber:.3e}")

# ============================================================
# VALIDATION
# ============================================================

logger.info("\n" + "="*60)
logger.info("VALIDATING ML MODELS")
logger.info("="*60)

# Storage for results
actual_snr = []
predicted_snr = []
actual_ber = []
predicted_ber = []

n_samples = 5000
logger.info(f"\nRunning {n_samples} validation samples...")

for i in range(n_samples):
    # Generate random parameters (same ranges as your training data)
    wavelength = np.random.uniform(1525, 1575)
    power = np.random.uniform(-10, 10)
    temp = np.random.uniform(20, 60)
    mod = np.random.choice(["NRZ", "PAM4", "QAM16"])
    
    # Get ACTUAL result from simulation (ground truth)
    sim_result = simulate_channel(wavelength, power, temp, mod, noise_std_dB=1.0)
    
    # Get PREDICTED result from ML model
    pred_snr, pred_ber = predict_performance(wavelength, power, temp, mod)
   

    # Store both
    actual_snr.append(sim_result['snr_dB'])
    predicted_snr.append(pred_snr)
    actual_ber.append(sim_result['ber'])
    predicted_ber.append(pred_ber)

    
    # Progress indicator every 1000 samples
    if (i + 1) % 1000 == 0:
        logger.info(f"Completed {i + 1}/{n_samples} samples")

logger.info("Validation complete!")

print(sim_result)
print("predict_snr", pred_snr)
print("predict_ber",pred_ber)


# ============================================================
# VALIDATION RESULTS
# ============================================================

logger.info("\n" + "="*60)
logger.info("VALIDATION RESULTS")
logger.info("="*60)

# Calculate metrics for SNR predictions
rmse_snr = root_mean_squared_error(actual_snr, predicted_snr)
r2_snr = r2_score(actual_snr, predicted_snr)

logger.info(f"\nSNR Model Performance:")
logger.info(f"  RMSE: {rmse_snr:.3f} dB")
logger.info(f"  R² Score: {r2_snr:.4f}")

# ============================================================
# VALIDATION PLOTS
# ============================================================

logger.info("\n" + "="*60)
logger.info("CREATING VALIDATION PLOTS")
logger.info("="*60)

# Create figure with 2 subplots
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# ====== PLOT 1: SNR Predictions ======
ax.scatter(actual_snr, predicted_snr, alpha=0.5, s=10, color='blue')
min_snr = min(min(actual_snr), min(predicted_snr))
max_snr = max(max(actual_snr), max(predicted_snr))
ax.plot([min_snr, max_snr], [min_snr, max_snr], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Actual SNR (dB)', fontsize=12)
ax.set_ylabel('Predicted SNR (dB)', fontsize=12)
ax.set_title(f'SNR Predictions (R²={r2_snr:.4f})', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
validation_plot = os.path.join(FIGURES_DIR, f'validation_results_{TIMESTAMP}.png')
plt.savefig(validation_plot, dpi=300, bbox_inches='tight')
logger.info(f"Plot saved to {validation_plot}")
plt.close()

# ============================================================
# BAYESIAN OPTIMIZATION
# ============================================================

logger.info("\n" + "="*60)
logger.info("BAYESIAN OPTIMIZATION")
logger.info("="*60)

# Fixed temperature for optimization
FIXED_TEMPERATURE = 40  # degrees C

def optimization_objective(params):
    """
    Objective function to MAXIMIZE SNR
    Bayesian optimization minimizes, so we return negative SNR
    """
    wavelength_nm, launch_power_dBm, mod_format = params
    
    # Encode modulation
    mod_encoded = encode_modulation(mod_format)
    
    # Create feature DataFrame
    X = pd.DataFrame([{
        "wavelength_nm": wavelength_nm,
        "launch_power_dBm": launch_power_dBm,
        "temperature_C": FIXED_TEMPERATURE,
        **mod_encoded
    }])
    
    # Predict SNR using ML model
    predicted_snr = snr_model.predict(X)[0]
    
    # Add penalty for excessive power (realistic constraint)
    # High power can cause nonlinear effects in real fibers
    if launch_power_dBm > 5:
        penalty = 0.5 * (launch_power_dBm - 5)**2
        predicted_snr -= penalty
    
    # Return negative SNR (because optimizer minimizes)
    return -predicted_snr

logger.info(f"Optimization target: Maximize SNR")
logger.info(f"Fixed temperature: {FIXED_TEMPERATURE}°C")
logger.info(f"Power penalty: Applied above +5 dBm")

# Define parameter search space
search_space = [
    Real(1530.0, 1570.0, name="wavelength_nm"),      # Wavelength range
    Real(-15.0, 10.0, name="launch_power_dBm"),      # Power range
    Categorical(["NRZ", "PAM4", "QAM16"], name="mod_format"),  # Modulation options
]

logger.info("\nSearch space:")
logger.info("  Wavelength: 1530-1570 nm")
logger.info("  Power: -15 to +10 dBm")
logger.info("  Modulation: NRZ, PAM4, QAM16")

logger.info("\nRunning Bayesian Optimization...")
logger.info("This will take about 30-60 seconds...")

n_calls = 50  # Number of optimization iterations

result = gp_minimize(
    func=optimization_objective,
    dimensions=search_space,
    n_calls=n_calls,
    n_initial_points=15,
    random_state=42,
    verbose=False
)

logger.info("Optimization complete!")

# Extract optimal parameters
best_wavelength = result.x[0]
best_power = result.x[1]
best_modulation = result.x[2]
best_neg_snr = result.fun
best_snr = -best_neg_snr  # Convert back to positive

logger.info("\n" + "="*60)
logger.info("OPTIMAL PARAMETERS FOUND")
logger.info("="*60)
logger.info(f"  Wavelength:    {best_wavelength:.2f} nm")
logger.info(f"  Launch Power:  {best_power:.2f} dBm")
logger.info(f"  Modulation:    {best_modulation}")
logger.info(f"  Expected SNR:  {best_snr:.2f} dB")
logger.info("="*60)

# ============================================================
# OPTIMIZATION CONVERGENCE PLOTS
# ============================================================

logger.info("\nGenerating optimization convergence plot...")

fig = plt.figure(figsize=(12, 5))

# Plot 1: Convergence
ax1 = plt.subplot(1, 2, 1)
plot_convergence(result)
plt.title("Bayesian Optimization Convergence", fontsize=13, fontweight='bold')

# Plot 2: Iteration values
ax2 = plt.subplot(1, 2, 2)
iterations = range(len(result.func_vals))
# Convert negative SNR back to positive for plotting
snr_values = [-val for val in result.func_vals]
ax2.plot(iterations, snr_values, marker='o', linestyle='-', alpha=0.7, markersize=4)
ax2.axhline(y=best_snr, color='red', linestyle='--', linewidth=2,
           label=f'Best SNR: {best_snr:.2f} dB')
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('Predicted SNR (dB)', fontsize=12)
ax2.set_title('SNR per Iteration', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

plt.tight_layout()
convergence_plot = os.path.join(FIGURES_DIR, f'optimization_convergence_{TIMESTAMP}.png')
plt.savefig(convergence_plot, dpi=300, bbox_inches='tight')
logger.info(f"Convergence plot saved to {convergence_plot}")
plt.close()

# ============================================================
# BEFORE vs AFTER COMPARISON
# ============================================================

logger.info("\n" + "="*60)
logger.info("BEFORE vs AFTER COMPARISON")
logger.info("="*60)

# Define baseline configuration (typical starting point)
baseline_config = {
    'wavelength_nm': 1550.0,
    'launch_power_dBm': 0.0,
    'temperature_C': FIXED_TEMPERATURE,
    'mod_format': 'PAM4'
}

# Define optimized configuration
optimized_config = {
    'wavelength_nm': best_wavelength,
    'launch_power_dBm': best_power,
    'temperature_C': FIXED_TEMPERATURE,
    'mod_format': best_modulation
}

# Simulate BOTH configurations (multiple runs to average out noise)
logger.info("\nSimulating baseline configuration (10 runs)...")
baseline_snr_list = []
baseline_ber_list = []
for i in range(10):
    result_sim = simulate_channel(**baseline_config, noise_std_dB=1.0)
    baseline_snr_list.append(result_sim['snr_dB'])
    baseline_ber_list.append(result_sim['ber'])

baseline_snr_avg = np.mean(baseline_snr_list)
baseline_ber_avg = np.mean(baseline_ber_list)

logger.info("Simulating optimized configuration (10 runs)...")
optimized_snr_list = []
optimized_ber_list = []
for i in range(10):
    result_sim = simulate_channel(**optimized_config, noise_std_dB=1.0)
    optimized_snr_list.append(result_sim['snr_dB'])
    optimized_ber_list.append(result_sim['ber'])

optimized_snr_avg = np.mean(optimized_snr_list)
optimized_ber_avg = np.mean(optimized_ber_list)

# Print comparison
logger.info("\n" + "="*60)
logger.info("BASELINE Configuration:")
logger.info("="*60)
logger.info(f"  Wavelength:  {baseline_config['wavelength_nm']:.2f} nm")
logger.info(f"  Power:       {baseline_config['launch_power_dBm']:.2f} dBm")
logger.info(f"  Modulation:  {baseline_config['mod_format']}")
logger.info(f"  Temperature: {baseline_config['temperature_C']:.1f}°C")
logger.info(f"  Avg SNR:     {baseline_snr_avg:.2f} dB")
logger.info(f"  Avg BER:     {baseline_ber_avg:.3e}")

logger.info("\n" + "="*60)
logger.info("OPTIMIZED Configuration:")
logger.info("="*60)
logger.info(f"  Wavelength:  {optimized_config['wavelength_nm']:.2f} nm")
logger.info(f"  Power:       {optimized_config['launch_power_dBm']:.2f} dBm")
logger.info(f"  Modulation:  {optimized_config['mod_format']}")
logger.info(f"  Temperature: {optimized_config['temperature_C']:.1f}°C")
logger.info(f"  Avg SNR:     {optimized_snr_avg:.2f} dB")
logger.info(f"  Avg BER:     {optimized_ber_avg:.3e}")

# Calculate improvement
snr_improvement = optimized_snr_avg - baseline_snr_avg
ber_improvement_factor = baseline_ber_avg / max(optimized_ber_avg, 1e-15)

logger.info("\n" + "="*60)
logger.info("IMPROVEMENT:")
logger.info("="*60)
logger.info(f"  SNR Gain:         {snr_improvement:+.2f} dB ({abs(snr_improvement/baseline_snr_avg*100):.1f}% improvement)")
logger.info(f"  BER Improvement:  {ber_improvement_factor:.2e}× better")
logger.info("="*60)

# ============================================================
# COMPARISON PLOTS
# ============================================================

logger.info("\nGenerating before/after comparison plot...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Wavelength
axes[0, 0].bar(['Baseline', 'Optimized'], 
              [baseline_config['wavelength_nm'], optimized_config['wavelength_nm']],
              color=['#3498db', '#2ecc71'], edgecolor='black', linewidth=1.5)
axes[0, 0].set_ylabel('Wavelength (nm)', fontsize=12)
axes[0, 0].set_title('Wavelength Comparison', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Plot 2: Launch Power
axes[0, 1].bar(['Baseline', 'Optimized'], 
              [baseline_config['launch_power_dBm'], optimized_config['launch_power_dBm']],
              color=['#3498db', '#2ecc71'], edgecolor='black', linewidth=1.5)
axes[0, 1].set_ylabel('Launch Power (dBm)', fontsize=12)
axes[0, 1].set_title('Launch Power Comparison', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')#3498db #2ecc71

# Plot 3: SNR
axes[1, 0].bar(['Baseline', 'Optimized'], 
              [baseline_snr_avg, optimized_snr_avg],
              color=['#e74c3c', '#2ecc71'], edgecolor='black', linewidth=1.5)
axes[1, 0].set_ylabel('SNR (dB)', fontsize=12)
axes[1, 0].set_title(f'SNR: {snr_improvement:+.2f} dB improvement', 
                     fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')
# Add value labels on bars
for i, v in enumerate([baseline_snr_avg, optimized_snr_avg]):
    axes[1, 0].text(i, v + 1, f'{v:.1f} dB', ha='center', fontweight='bold')

# Plot 4: Modulation Format
mod_comparison = [baseline_config['mod_format'], optimized_config['mod_format']]
axes[1, 1].bar(['Baseline', 'Optimized'], [1, 1], 
              color=['#3498db', '#2ecc71'], edgecolor='black', linewidth=1.5)
axes[1, 1].set_ylim(0, 1.5)
axes[1, 1].set_ylabel('Modulation Format', fontsize=12)
axes[1, 1].set_title('Modulation Format', fontsize=13, fontweight='bold')
axes[1, 1].set_yticks([])
# Add text labels
axes[1, 1].text(0, 0.5, mod_comparison[0], ha='center', va='center', 
               fontsize=16, fontweight='bold')
axes[1, 1].text(1, 0.5, mod_comparison[1], ha='center', va='center', 
               fontsize=16, fontweight='bold')

plt.suptitle('Baseline vs Optimized Configuration', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
comparison_plot = os.path.join(FIGURES_DIR, f'before_after_comparison_{TIMESTAMP}.png')
plt.savefig(comparison_plot, dpi=300, bbox_inches='tight')``
logger.info(f"Comparison plot saved to {comparison_plot}")
plt.close()

# ============================================================
# SUMMARY
# ============================================================

logger.info("\n" + "="*60)
logger.info("PIPELINE COMPLETE!")
logger.info("="*60)
logger.info(f"\nGenerated outputs for run {TIMESTAMP}:")
logger.info(f"  1. {validation_plot}")
logger.info(f"  2. {convergence_plot}")
logger.info(f"  3. {comparison_plot}")
logger.info(f"  4. {log_filename}")
logger.info("\nAll steps completed successfully!")
logger.info("="*60)