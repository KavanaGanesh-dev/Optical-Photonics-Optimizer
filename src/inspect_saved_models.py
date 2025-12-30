import pickle
import pandas as pd

# Load SNR model
with open("/Users/kavanakiran/Documents/AI_Projects/optical-photonics-optimizer/models/snr_model.pkl", "rb") as f:
    snr_model = pickle.load(f)


#  I am checking here if the model has been loaded in the pickle file
print("Model type:")
print(type(snr_model))

# Lets print and see the methods : Basically I am query the model here
print("\nModel parameters:")
print(snr_model.get_params())


feature_names = [
    "wavelength_nm",
    "launch_power_dBm",
    "temperature_C",
    "mod_format_NRZ",
    "mod_format_PAM4",
    "mod_format_QAM16",
]

print("\nFeature importance:")
for name, importance in zip(feature_names, snr_model.feature_importances_):
    print(f"{name:} : {importance:.3f}")



# Create one test sample
X_test = pd.DataFrame([{
    "wavelength_nm": 1550,
    "launch_power_dBm": 0,
    "temperature_C": 25,
    "mod_format_NRZ": 1,
    "mod_format_PAM4": 0,
    "mod_format_QAM16": 0,
}])

predicted_snr = snr_model.predict(X_test)

print("Predicted SNR for test sample:")
print(predicted_snr)