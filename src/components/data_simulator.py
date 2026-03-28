import numpy as np
import batman
import matplotlib.pyplot as plt
from tqdm import tqdm

class TransitSimulator:
    def __init__(self, time_array):
        """
        Initializes the simulator with a specific time observation window.
        """
        self.time = time_array
        
    def generate_spherical_transit(self, rp_rs, period, a_rs, inc):
        """
        Generates a standard spherical transit light curve using batman.
        """
        params = batman.TransitParams()
        params.t0 = 0.0                       
        params.per = period                   
        params.rp = rp_rs                     
        params.a = a_rs                       
        params.inc = inc                      
        params.ecc = 0.0                      
        params.w = 90.0                       
        params.u = [0.1, 0.3]                 
        params.limb_dark = "quadratic"        

        m = batman.TransitModel(params, self.time)
        return m.light_curve(params)
        
    def inject_j2_anomaly(self, base_flux, j2_value):
        """
        Injects a mock J2 (oblateness) perturbation.
        """
        gradient = np.gradient(base_flux)
        anomaly = j2_value * np.abs(gradient) 
        return base_flux - anomaly

    def add_red_noise(self, flux, noise_level=100e-6):
        """
        Adds correlated (red) noise to simulate stellar variability.
        """
        red_noise = np.zeros_like(flux)
        red_noise[0] = np.random.normal(0, noise_level)
        for i in range(1, len(flux)):
            red_noise[i] = 0.8 * red_noise[i-1] + np.random.normal(0, noise_level * 0.2)
        return flux + red_noise

    def generate_lightcurve(self, rp_rs, period, a_rs, inc, j2, noise_level=100e-6):
        """
        Full pipeline: Spherical Transit -> J2 Injection -> Noise Injection
        """
        clean_flux = self.generate_spherical_transit(rp_rs, period, a_rs, inc)
        oblate_flux = self.inject_j2_anomaly(clean_flux, j2)
        noisy_flux = self.add_red_noise(oblate_flux, noise_level)
        return clean_flux, oblate_flux, noisy_flux

    def generate_dataset(self, num_samples=10000, noise_level=150e-6):
        """
        Mass-generates a dataset of light curves by randomizing planetary parameters.
        Returns X (features/flux) and y (targets/J2).
        """
        X = np.zeros((num_samples, len(self.time)))
        y = np.zeros(num_samples)
        
        print(f"Generating {num_samples} synthetic light curves...")
        for i in tqdm(range(num_samples)):
            # Randomize physical parameters within realistic ranges
            rp_rs = np.random.uniform(0.05, 0.15)  # Planet radius
            period = np.random.uniform(2.0, 10.0)  # Orbital period
            a_rs = np.random.uniform(5.0, 15.0)    # Semi-major axis
            inc = np.random.uniform(87.0, 90.0)    # Inclination
            
            # The target variable: J2
            j2 = np.random.uniform(0.0, 0.05)      
            
            _, _, noisy_flux = self.generate_lightcurve(rp_rs, period, a_rs, inc, j2, noise_level)
            
            X[i] = noisy_flux
            y[i] = j2
            
        return X, y