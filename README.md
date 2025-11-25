Create a simplified channel model with the following parameters
    1. Loss
    2. Attenuation 
    3. Modulation 
    4. Noise
    5. Mapping SNR to BER

Notes : Silicon Waveguide are optimized near 1550nm. If wavelength more than this - more loss
This target wavelength of 1550 nm is significant in fiber optics as it is a standard wavelength for long-distance telecommunications due to low signal attenuation in optical fibers.

SNR = received_power - noise_floor (Receiver noise floor is fixed)
received_power = launch_power - total_loss
total_loss = wavelength_loss + temp_penalty
temp_penalty = (temperature - 25°C) × constant 
Higher temperature means more scattering

loss = base_loss + penalty_based_on_how_far_wavelength_is_from_1550


NRZ needs less SNR
PAN needs more SNR
QAM16 needs mouch more
effective_SNR = SNR - modulation_penalty

Optical system has noise, jitter and drift
effective_SNR_noisy = effective_SNR + random_noise_value

'''BER from SNR: High SNR means very low BER. Low SNR means very bad SNR'''
BER = 0.5 × erfc( sqrt(SNR_linear/2) )


1. .wavelength_loss_dB()
2. temperature_penalty_dB()
3. mod_format_snr_penalty_dB()
4. snr_dB_to_ber()
simulate_channel() - combine and simulating all of them

NRZ has 2 levels - low SNR - Simple, big eye opening
PAM4 has 4 levels - medium - Denser levels - more noise - sensitive
QAM16 has 16 levels - high SNR - very tight spacing betweent the symbols

