import numpy as np
import lightkurve as lk
from pytransit import QuadraticModel
import warnings

# Suppress lightkurve's standard warnings for clean notebook output
warnings.filterwarnings("ignore", module="lightkurve")

class EmpiricalSimulator:
    def __init__(self):
        # Initialize the high-speed PyTransit Quadratic Limb Darkening model
        self.tm = QuadraticModel()
        self.real_time = None
        self.real_flux = None

    def fetch_tess_noise_floor(self, tic_id="TIC 270810595"):
        """
        Fetches REAL TESS photometric data from the MAST archive.
        Automatically finds the first available sector to prevent NoneType errors.
        """
        print(f"Searching MAST archive for {tic_id}...")
        
        # Search for all available sectors for this target
        search_result = lk.search_lightcurve(tic_id, mission='TESS')        
        if len(search_result) == 0:
            raise ValueError(f"MAST Archive error: No SPOC light curves found for {tic_id}.")
            
        print(f"Found {len(search_result)} available observations. Downloading the first one...")
        
        # Download the first available sector
        lc = search_result[0].download(download_dir='../data/raw')
        
        if lc is None:
            raise ValueError("Download failed. The MAST server might be timing out.")
            
        # Clean the light curve: remove NaNs, outliers, and flatten stellar variations
        print("Cleaning and flattening the stellar data...")
        clean_lc = lc.remove_nans().remove_outliers(sigma=5).flatten(window_length=101)
        
        self.real_time = clean_lc.time.value
        self.real_flux = clean_lc.flux.value
        
        print("Real TESS background noise successfully ingested.")
        return self.real_time, self.real_flux

    def generate_oblate_transit(self, time_array, rp_rs, period, a_rs, inc, j2):
        """
        Uses PyTransit to generate the base transit, then applies a rigorous 
        oblateness mathematical modifier to the ingress/egress slopes.
        """
        self.tm.set_data(time_array)
        
        # Standard stellar limb darkening coefficients for a Sun-like star
        ldc = np.array([0.4, 0.2]) 
        
        # Base spherical PyTransit model
        base_flux = self.tm.evaluate(k=rp_rs, ldc=ldc, t0=0.0, p=period, a=a_rs, i=inc)
        
        # Rigorous Oblateness Modification
        # An oblate planet alters the projected cross-sectional area dynamically 
        # during the transit boundaries (ingress/egress).
        gradient = np.gradient(base_flux)
        oblateness_effect = j2 * np.abs(gradient) 
        
        oblate_flux = base_flux - oblateness_effect
        return oblate_flux

    def get_local_view(self, period, rp_rs, a_rs, inc, j2):
        """
        Injects the signal into the real TESS data, then phase-folds it to 
        create the 'Local View Window' (as defined in your Tri-Input architecture).
        """
        if self.real_time is None:
            raise ValueError("Must run fetch_tess_noise_floor() first.")
            
        # 1. Generate the transit across the full 27-day TESS sector
        transit_signal = self.generate_oblate_transit(self.real_time, rp_rs, period, a_rs, inc, j2)
        
        # 2. Inject it into the real noise (multiplicative for normalized flux)
        synthetic_observation = self.real_flux * transit_signal
        
        # 3. Phase-fold the data to isolate the 'Local View'
        phase = (self.real_time % period)
        # Shift phase so transit is centered at 0
        phase[phase > period/2] -= period
        
        # 4. Sort by phase to create a smooth sequence for the 1D-CNN
        sort_idx = np.argsort(phase)
        folded_phase = phase[sort_idx]
        folded_flux = synthetic_observation[sort_idx]
        
        # 5. Crop to the local window (e.g., +/- 0.1 days from mid-transit)
        mask = (folded_phase >= -0.1) & (folded_phase <= 0.1)
        
        return folded_phase[mask], folded_flux[mask]
    def generate_dataset(self, num_samples=5000, n_features=250):
        """
        Mass-generates a dataset of light curves embedded in real TESS noise.
        Interpolates the empirical data to a fixed length (n_features) for the CNN.
        Returns X (flux features) and y (J2 targets).
        """
        if self.real_time is None:
            raise ValueError("Must fetch TESS noise floor first.")
            
        from tqdm import tqdm
        
        X = np.zeros((num_samples, n_features))
        y = np.zeros(num_samples)
        
        print(f"Generating {num_samples} samples using empirical TESS noise...")
        
        for i in tqdm(range(num_samples)):
            # 1. Randomize physical parameters within realistic ranges
            rp_rs = np.random.uniform(0.05, 0.15)  
            period = np.random.uniform(2.0, 5.0)   
            a_rs = np.random.uniform(5.0, 10.0)    
            inc = np.random.uniform(87.0, 90.0)    
            j2 = np.random.uniform(0.0, 0.05)      
            
            # 2. Generate the transit and inject it into the TESS static
            phase, flux = self.get_local_view(period, rp_rs, a_rs, inc, j2)
            
            # 3. Interpolate to a fixed length for PyTorch CNN
            # We map the chaotic, gappy TESS data onto a perfect 250-point grid
            fixed_phase = np.linspace(-0.1, 0.1, n_features)
            fixed_flux = np.interp(fixed_phase, phase, flux)
            
            X[i] = fixed_flux
            y[i] = j2
            
        return X, y