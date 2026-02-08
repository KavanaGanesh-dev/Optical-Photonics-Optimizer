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

    ber_clipped = np.clip(y_ber, 1e-12, 1.0) #Prepare log10 BER target (avoid log(0) by clipping)
    y_ber_log = np.log10(ber_clipped)

    # ADD THIS: Check the distribution of BER values
    print("\n" + "="*60)
    print("ANALYZING BER TARGET VALUES")
    print("="*60)
    print(f"Original BER range: {y_ber.min():.3e} to {y_ber.max():.3e}")
    print(f"Log BER range: {y_ber_log.min():.3f} to {y_ber_log.max():.3f}")
    print(f"Log BER mean: {y_ber_log.mean():.3f}")
    print(f"Log BER std: {y_ber_log.std():.3f}")
    print("\nLog BER distribution:")
    print(y_ber_log.describe())
    print("="*60 + "\n")


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

    rmse_ber_log = root_mean_squared_error(y_test_ber, y_pred_ber_log)
    r2_ber_log = r2_score(y_test_ber, y_pred_ber_log)
    print(f"BER Model RMSE (log scale): {rmse_ber_log:.3f}")
    print(f"BER Model R² (log scale): {r2_ber_log:.3f}")

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

    # ============================================================
    # CREATE PLOTS AT THE END (NON-BLOCKING)
    # ============================================================
    print("\nGenerating comparison plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ====== PLOT 1: SNR Predictions ======
    axes[0].scatter(y_test_snr, y_pred_snr, alpha=0.4, label="Random Forest", s=20)
    axes[0].scatter(y_test_snr, y_pred_snr_mlp, alpha=0.4, label="MLP", s=20)
    
    mn = min(y_test_snr.min(), y_pred_snr.min(), y_pred_snr_mlp.min())
    mx = max(y_test_snr.max(), y_pred_snr.max(), y_pred_snr_mlp.max())
    axes[0].plot([mn, mx], [mn, mx], "k--", linewidth=2, label="Ideal")
    
    axes[0].set_xlabel("Actual SNR (dB)", fontsize=12)
    axes[0].set_ylabel("Predicted SNR (dB)", fontsize=12)
    axes[0].set_title(f"SNR Prediction: RF vs MLP (R²={r2_snr:.3f})", fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ====== PLOT 2: BER Predictions ======
    axes[1].scatter(y_test_ber, y_pred_ber_log, alpha=0.5, s=20, color='red')
    
    mn_ber = min(y_test_ber.min(), y_pred_ber_log.min())
    mx_ber = max(y_test_ber.max(), y_pred_ber_log.max())
    axes[1].plot([mn_ber, mx_ber], [mn_ber, mx_ber], "k--", linewidth=2, label="Ideal")
    
    axes[1].set_xlabel("Actual log₁₀(BER)", fontsize=12)
    axes[1].set_ylabel("Predicted log₁₀(BER)", fontsize=12)
    axes[1].set_title(f"BER Prediction (R²={r2_ber_log:.3f})", fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../figures/training_results.png', dpi=300, bbox_inches='tight')
    print("✓ Training plots saved to figures/training_results.png")
    plt.show()  # Show at the very end
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
