import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# Define a utility function to clean column names
def clean_column_names(columns):
    cleaned_columns = [
        col.strip()
        .replace(' ', '_')
        .replace('/', '_')
        .replace('°', 'deg')
        .replace(',', '_')
        .replace('\x00', '')
        for col in columns
    ]
    return cleaned_columns

# Define the impedance model for the circuit: R -- (C || CPE)
def impedance_model_R_C_parallel_CPE(f, R, C, Q, n):
    omega = 2 * np.pi * f
    Y_C = 1j * omega * C  # Admittance of the capacitor
    Y_CPE = Q * (1j * omega) ** n  # Admittance of the CPE
    Y_parallel = Y_C + Y_CPE  # Total parallel admittance
    Z_parallel = 1 / Y_parallel  # Convert to impedance
    Z_total = R + Z_parallel  # Total impedance of the circuit
    return np.concatenate((Z_total.real, -Z_total.imag))

# Calculate goodness of fit
def calculate_r_squared(y_exp, y_fit):
    ss_res = np.sum((y_exp - y_fit) ** 2)
    ss_tot = np.sum((y_exp - np.mean(y_exp)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

# Directory containing the data for R2 D4
data_dir = "/"

fit_results = {}

# Process through set ranges, diameters, and lengths
for day in range(1, 8):
    for diameter in range(2,4):
        for length in range(1, 6):
            file_name = f"X_d{day}_D{diameter}_L{length}.csv"
            file_path = os.path.join(data_dir, f"day{day}", file_name)

            if os.path.exists(file_path):
                print(f"Processing file: {file_path}")
                try:
                    df = pd.read_csv(file_path, encoding='utf-16', on_bad_lines='skip', skiprows=5)
                except UnicodeDecodeError as e:
                    print(f"Error reading file {file_path}: {e}")
                    continue

                df.columns = clean_column_names(df.columns)
                df_cleaned = df.dropna(axis=1, how='all')

                df_cleaned['freq___Hz'] = pd.to_numeric(df_cleaned['freq___Hz'], errors='coerce')
                df_cleaned["Z'___Ohm"] = pd.to_numeric(df_cleaned["Z'___Ohm"], errors='coerce')

                if "Z''___Ohm" in df_cleaned.columns:
                    df_cleaned["Z''___Ohm"] = pd.to_numeric(df_cleaned["Z''___Ohm"], errors='coerce')
                    df_cleaned["Z''___Ohm"] = np.abs(df_cleaned["Z''___Ohm"])
                elif "Z___Ohm" in df_cleaned.columns:
                    df_cleaned["Z___Ohm"] = pd.to_numeric(df_cleaned["Z___Ohm"], errors='coerce')
                    df_cleaned["Z''___Ohm"] = np.abs(df_cleaned["Z___Ohm"])

                df_cleaned = df_cleaned.dropna()
                
                frequencies = df_cleaned['freq___Hz'].values
                Z_real_exp = df_cleaned["Z'___Ohm"].values
                Z_imag_exp = df_cleaned["Z''___Ohm"].values
                Z_exp_combined = np.concatenate((Z_real_exp, Z_imag_exp))
                
                # Initial guess for R -- (C || CPE) parameters: R, C, Q, n
                initial_guess = [34000, 0.947e-9, 0.1e-6, 0.6]  # R, C, Q, n
                bounds = ([3.2e4, 1e-13, 2e-10, 0.5], [1e7, 1e-5, 2e-5, 1])
                
                try:
                    # Fit the model to the data
                    popt, _ = curve_fit(lambda f, R, C, Q, n: impedance_model_R_C_parallel_CPE(f, R, C, Q, n),
                                        frequencies, Z_exp_combined, p0=initial_guess, bounds=bounds, method='trf')

                    R_fit, C_fit, Q_fit, n_fit = popt
                    Z_fit = impedance_model_R_C_parallel_CPE(frequencies, R_fit, C_fit, Q_fit, n_fit)
                    Z_fit_real = Z_fit[:len(frequencies)]
                    Z_fit_imag = Z_fit[len(frequencies):]
                    
                    r_squared_real = calculate_r_squared(Z_real_exp, Z_fit_real)
                    r_squared_imag = calculate_r_squared(Z_imag_exp, Z_fit_imag)
                    r_squared_overall = calculate_r_squared(Z_exp_combined, np.concatenate((Z_fit_real, Z_fit_imag)))
                    
                    # Store fitting results
                    fit_results[f"Day {day} Length {length} D{diameter}"] = {
                        "R": R_fit, "C": C_fit, "Q": Q_fit, "n": n_fit,
                        "R^2 Real": r_squared_real,
                        "R^2 Imag": r_squared_imag,
                        "R^2 Overall": r_squared_overall
                    }

                    # Plot the Nyquist plot
                    plt.figure(figsize=(10, 5))
                    plt.plot(Z_real_exp, Z_imag_exp, 'o', label="Experimental Data")
                    plt.plot(Z_fit_real, Z_fit_imag, '-', label="Fitted Model")
                    plt.xlabel("Real(Z) [Ohms]")
                    plt.ylabel("Imag(Z) [Ohms]")
                    plt.legend()
                    plt.title(f"Nyquist Plot - Day {day}, Length {length} D4")
                    plt.grid()
                    plt.show()
                
                except RuntimeError as e:
                    print(f"Fitting failed for Day {day}, Length {length} D4: {e}")

# Save fitting results to a CSV file
fit_results_df = pd.DataFrame.from_dict(fit_results, orient='index')
fit_results_df.index.name = "Dataset"
fit_results_df.to_csv(os.path.join(data_dir, "fitting_results_B1.csv"), mode='w', header=True, index=True)
print("Fitting results saved.")
