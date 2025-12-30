import pandas as pd #loading the csv into a DataFrame
import numpy as np #for numerical operations

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split    #split data into train and test
from sklearn.ensemble import RandomForestRegressor  #ML model
from sklearn.metrics import root_mean_squared_error, r2_score    #evaluation metrics
import matplotlib.pyplot as plt
import pickle   #save trained models to disk

def main():
    df = pd.read_csv("../data/optical_channel_dataset.csv")
    print(df.head())
    # To remember: The imputs: 'wavelength_nm', 'launch_power_dBm', 'temperature_C', 'mod_format'
            #  The outputs: 'snr_dB', 'ber'
    print(df. columns.tolist()) # to see the headers
    print(df.shape) # to see the rows, columns


    #Converting the mod_format text into numerical ones
    df_encoded = pd.get_dummies(df, columns=['mod_format'])
    print("\n columns after encoding:")
    print(df_encoded.columns)
    print(df.shape) # to see the rows, columns

    # These are X features
    feature_cols = ['wavelength_nm', 'launch_power_dBm', 'temperature_C',
       'mod_format_NRZ', 'mod_format_PAM4', 'mod_format_QAM16']  
    X = df_encoded[feature_cols]

    # These are Y features
    y_snr = df_encoded["snr_dB"]
    y_ber = df_encoded["ber"]

    print("\n X Feature matrix shape", X.shape)
    print("\n SNR target shape ", y_snr.shape)
    print("\n BER target shape ", y_ber.shape)


    #spliitng the data : Training set is 80% and test set is 20%
    # Train Test split for SNR model
    X_train_snr, X_test_snr, y_train_snr, y_test_snr = train_test_split(X, y_snr, test_size=0.2, random_state=42)


    print("\nTrain size (SNR):", X_train_snr.shape[0])
    print("\nTest size (SNR):", X_test_snr.shape[0])

    # print("\nTrain size (SNR):", y_train_snr.shape[0])
    # print("\nTest size (SNR):", y_test_snr.shape[0])


    #Train Random Forest Regressor ML model for SNR
    snr_model = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
    print("\nTraining SNR model..")
    snr_model.fit(X_train_snr, y_train_snr)

    # Evaluating the SNR model based on RMSE and R^2
    y_pred_snr = snr_model.predict(X_test_snr)
    rmse_snr = root_mean_squared_error(y_test_snr, y_pred_snr) #error in dB
    r2_snr = r2_score(y_test_snr, y_pred_snr) #variance

    print(f"SNR model RMSE: {rmse_snr:.3f} dB")
    print(f"SNR model R2: {r2_snr:.3f}")


# Now i want to implement MLP model for SNR
    print("\nTraining MLP model for SNR...")

    mlp_snr = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
    )

    mlp_snr.fit(X_train_snr, y_train_snr)

    y_pred_snr_mlp = mlp_snr.predict(X_test_snr)

    rmse_snr_mlp = root_mean_squared_error(y_test_snr, y_pred_snr_mlp)
    r2_snr_mlp = r2_score(y_test_snr, y_pred_snr_mlp)

    print(f"MLP SNR RMSE: {rmse_snr_mlp:.3f} dB")
    print(f"MLP SNR R2: {r2_snr_mlp:.3f}")

    # Actual vs predicted snr

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_snr, y_pred_snr, alpha=0.4, label="Random Forest")
    plt.scatter(y_test_snr, y_pred_snr_mlp, alpha=0.4, label="MLP")

    mn = min(y_test_snr.min(), y_pred_snr.min(), y_pred_snr_mlp.min())
    mx = max(y_test_snr.max(), y_pred_snr.max(), y_pred_snr_mlp.max())

    plt.plot([mn, mx], [mn, mx], "k--", label="Ideal")



    plt.xlabel("Actual SNR (dB)")
    plt.ylabel("Predicted SNR (dB)")
    plt.title("SNR Prediction: RF vs MLP")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



    ber_clipped = np.clip(y_ber, 1e-12, 1.0) #Prepare log10 BER target (avoid log(0) by clipping)
    y_ber_log = np.log10(ber_clipped)


    X_train_ber, X_test_ber, y_train_ber, y_test_ber = train_test_split(
        X, y_ber_log, test_size=0.2, random_state=42
    ) #Train-test split for BER model (reuse X, new y)

    print("\nTrain size (BER):", X_train_ber.shape[0])
    print("Test size (BER):", X_test_ber.shape[0])

    # Train Random Forest model for log10(BER)
    ber_model = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42,n_jobs=-1,)

    print("\nTraining BER model (log10 scale)...")
    ber_model.fit(X_train_ber, y_train_ber)

    #Evaluate BER model
    y_pred_ber_log = ber_model.predict(X_test_ber)

    # convert back to linear BER just for interpretation
    y_pred_ber = 10 ** y_pred_ber_log
    y_true_ber = 10 ** y_test_ber

    mae_ber = np.mean(np.abs(y_pred_ber - y_true_ber))
    print(f"BER Model MAE (linear scale): {mae_ber:.3e}")

#   Save models to disk
    with open("../models/snr_model.pkl", "wb") as f:
        pickle.dump(snr_model, f)

    with open("../models/ber_model_log.pkl", "wb") as f:
        pickle.dump(ber_model, f)

    print("\n Both Models Ber model and SNR model stored in models folder")
if __name__ == "__main__":
    main()