from channel_model import wavelength_loss_dB
from channel_model import temperature_loss_dB
from channel_model import modulation_snr_loss_dB
from channel_model import snr_dB_to_ber
from channel_model import simulate_channel
 
print("Wavelength losses")
print(wavelength_loss_dB(1550)) # expect ~3.0
print(wavelength_loss_dB(1560)) # slightly higher
print(wavelength_loss_dB(1530)) # higher
print('\n')


print("Temperature losses")
print(temperature_loss_dB(25)) # expect 0
print(temperature_loss_dB(35)) # expect 0.2
print(temperature_loss_dB(15)) # expect -0.2
print('\n')


print("SNR based modulation losses")
print(modulation_snr_loss_dB("NRZ")) # # expect 0.0
print(modulation_snr_loss_dB("PAM4")) # expect 4.5
print(modulation_snr_loss_dB("QAM16")) # expect 8.0
print('\n')

print("SNR to ber ")
print((snr_dB_to_ber(20))) # very small BER (~10^-10 or smaller) - clean link no error
print((snr_dB_to_ber(10)))  # moderate BER (~10^-5 or so) - good link
print((snr_dB_to_ber(0))) # high BER (~0.07) - bad - errors
print((snr_dB_to_ber(-5))) # terrible BER (~0.15 to 0.3) - very noisy
print('\n')

print("simulating the channel ")
print(simulate_channel(wavelength_nm=1550 ,launch_power_dBm=0,temperature_C=25,mod_format="NRZ",noise_std_dB=1.0,)) # Standard transmitter power,launch_power=0dBm
print(simulate_channel(wavelength_nm=1550 ,launch_power_dBm=5,temperature_C=25,mod_format="NRZ",noise_std_dB=1.0,)) # high power lazer,launch_power=5dBm
print(simulate_channel(wavelength_nm=1550 ,launch_power_dBm=-10,temperature_C=25,mod_format="NRZ",noise_std_dB=1.0,)) # Very weak Tx, high BER expected,launch_power=-10dBm
