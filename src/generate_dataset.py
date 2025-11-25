import numpy as np #to genreate the random valiues for wavelength, power, temperature
import pandas as pd #save the data into csv file
from tqdm import tqdm # iterable that shows real time progress 
from channel_model import simulate_channel

def generate_dataset(n_samples=5000, path= "/Users/kavanakiran/Documents/AI_Projects/optical-photonics-optimizer/data/optical_channel_dataset.csv"):
    records = []

    for i in tqdm(range(n_samples), desc='simulating data for optical channel'):
        wavelength = np.random.uniform(1525, 1575)  #wavelngth in nm
        launch_power = np.random.uniform(-10,10)    #power in dBm
        temperature = np.random.uniform(20,60)      #temp in celsius
        mod_format = np.random.choice(["NRZ", "PAM4", "QAM16"]) # type: ignore


        sample = simulate_channel(wavelength_nm=wavelength, launch_power_dBm=launch_power,temperature_C=temperature, mod_format=mod_format )
        records.append(sample)

    df = pd.DataFrame(records)
    df.to_csv(path, index=False)

    print(f'Dataset saved to the {path}')


if __name__ == "__main__":
    generate_dataset()