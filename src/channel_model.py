import numpy as np
from scipy.special import erfc
#erfc() method returns the complementary error function 
# This method accepts a value between -inf and +inf and returns a floating-point value between 0 and 2.

# === Constants ===
REF_WAVELENGTH_NM = 1550.0       # Reference wavelength
BASE_LOSS_DB = 3.0               # Base insertion loss in dB
TEMP_COEFF = 0.02                # dB penalty per °C above 25
NOISE_FLOOR_DBM = -40.0          # Receiver noise floor


# ==== Defining the LOSS MODELS====
def wavelength_loss_dB(wavelength_nm:float) -> float:
    """
    This function returns oprtical loss in dB" depending 
    how far the length is from the optimal 1550nm point. 
    """
    detuning = (wavelength_nm - 1550.0) / 10.0 # how far away the current wavelngth from the optimal
    loss = 0.2 * detuning**2 # we want the penality/loss to be 0 or positive - that is why quadratic (0.2 scaling factor for smooth performance)
    return BASE_LOSS_DB + loss



def temperature_loss_dB(temperature_C: float) -> float:
    """
    Returns Temperature-related loss.
    25C is the baseline(0dB)
    Above 25C more loss
    Below 25C is less loss
    """
    return TEMP_COEFF * (temperature_C - 25.0)


def modulation_snr_loss_dB(modulation_method: str) -> float:
    """
    Returns SNR loss based on modulation format.
    Higher-order modulations need more SNR.
    NRZ : 0 dB
    PAM4: 4.5 dB penalty
    QAM16: 8 dB penalty
    """
    mod_format = modulation_method.upper()
    if mod_format == "NRZ":
        return 0.0
    if mod_format == "PAM4":
        return 4.5
    if mod_format == "QAM16":
        return 8.0
    return 2.0



def snr_dB_to_ber(snr_dB: float) -> float:
    """
    Convert SNR in dB to BER using AWGN approximation (Additive white Gaussian Noise)
    Higher SNR => lower BER.
    """
    # Convert SNR to linear scale
    snr_linear = 10 ** (snr_dB / 10.0)
    # Usinf AWGN BER for NRZ
    ber = 0.5 * erfc(np.sqrt(snr_linear / 2.0))
    return ber



def simulate_channel(
    wavelength_nm: float,
    launch_power_dBm: float,
    temperature_C: float,
    mod_format: str = "NRZ",
    noise_std_dB: float = 1.0,
):
    """
    Simulates a simplified optical channel and returns
    SNR and BER values along with the inputs.
    """
    # ===== 1. Compute wavelength-based loss =====
    loss_wavelength = wavelength_loss_dB(wavelength_nm)

    # ===== 2. Compute temperature-based penalty =====
    loss_temperature = temperature_loss_dB(temperature_C)

    # ===== 3. Total optical loss =====
    total_loss_dB = loss_wavelength + loss_temperature

    # ===== 4. Received power =====
    # launch_power_dBm is the TRansmitter power launch into the optical link
    # IMPORTANT : Higher launch power → higher received power → higher SNR → lower BER
    recv_power_dBm = launch_power_dBm - total_loss_dB

    # ===== 5. Baseline SNR (no modulation penalty yet) =====
    snr_dB = recv_power_dBm - (-40.0)   # noise floor assumed -40 dBm

    # ===== 6. Apply modulation penalty =====
    snr_dB -= modulation_snr_loss_dB(mod_format)

        # ===== 7. Add random channel variation =====
    snr_dB_noisy = snr_dB + np.random.normal(0.0, noise_std_dB)

    # ===== 8. Convert SNR -> BER =====
    ber = snr_dB_to_ber(snr_dB_noisy)
    
    # ===== 9. Return sample =====
    return {
        "wavelength_nm": wavelength_nm,
        "launch_power_dBm": launch_power_dBm,
        "temperature_C": temperature_C,
        "mod_format": mod_format,
        "snr_dB": snr_dB_noisy,
        "ber": float(ber),
    }
