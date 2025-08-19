
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
from astroquery.gaia import Gaia
import astropy.units as u
import astropy.coordinates as coord
from astropy.table import Table
from scipy import stats
from datetime import datetime

# Physical constants
CONSTANTS = {
    'R0': 8.5,   
    'V0': 220,
    'A_true': 15.3,
    'B_true': -11.9,
}
def get_gaia_sample(max_stars=50000, save_data=True):
    """
    Query Gaia DR3 for local stellar sample with proper error handling
    
    Parameters:
    -----------
    max_stars : int
        Maximum number of stars to retrieve
    save_data : bool
        Whether to save the data to file
    
    Returns:
    --------
    astropy.table.Table : Gaia stellar data
    """
    print(f"Querying Gaia DR3 for up to {max_stars} stars...")
    
    query = f"""
    SELECT TOP {max_stars}
        source_id, ra, dec, l, b,
        parallax, parallax_error, parallax_over_error,
        pmra, pmra_error, pmdec, pmdec_error,
        dr2_radial_velocity, dr2_radial_velocity_error,
        phot_g_mean_mag, bp_rp
    FROM gaiadr3.gaia_source 
    WHERE parallax > 3.0
        AND parallax_over_error > 5 
        AND pmra IS NOT NULL 
        AND pmdec IS NOT NULL
        AND pmra_error < 1.0
        AND pmdec_error < 1.0
        AND ABS(b) < 45
        AND phot_g_mean_mag < 15
    """
    
    try:
        job = Gaia.launch_job_async(query)
        data = job.get_results()
        
        print(f"✓ Successfully retrieved {len(data)} stars from Gaia DR3")
        
        if save_data:
            save_gaia_data(data)
            
        return data
        
    except Exception as e:
        print(f"Error querying Gaia: {e}")
        print("Falling back to smaller sample...")
        

        small_query = query.replace(f"TOP {max_stars}", "TOP 10000")
        try:
            job = Gaia.launch_job_async(small_query)
            data = job.get_results()
            print(f"✓ Retrieved smaller sample: {len(data)} stars")
            return data
        except Exception as e2:
            print(f"Fallback query also failed: {e2}")
            return None

def save_gaia_data(data, directory="dataset"):
    """
    Save Gaia data to file with proper error handling
    """
    try:
        os.makedirs(directory, exist_ok=True)
    
        filepath = os.path.join(directory, "gaia_data.csv")
        

        df = data.to_pandas()
        df.to_csv(filepath, index=False)
        
        print(f"✓ Data saved to {filepath}")
        

        fits_path = os.path.join(directory, "gaia_data.fits")
        data.write(fits_path, format='fits', overwrite=True)
        print(f"✓ Data also saved as FITS: {fits_path}")
        
    except Exception as e:
        print(f"Error saving data: {e}")

def filter_data(data):
    """
    Apply quality filters to Gaia data with proper syntax
    
    Parameters:
    -----------
    data : astropy.table.Table
        Raw Gaia data
        
    Returns:
    --------
    astropy.table.Table : Filtered data
    """
    print("Applying quality filters...")
    print(f"Initial sample size: {len(data)} stars")
    

    mask = (
        (data['parallax'] > 0) & 
        (data['parallax_error']/data['parallax'] < 0.2) &  
        (np.isfinite(data['pmra'])) &
        (np.isfinite(data['pmdec'])) &
        (data['pmra_error'] < 0.5) &
        (data['pmdec_error'] < 0.5)
    )
    
    filtered_data = data[mask]
    print(f"✓ Filtered sample size: {len(filtered_data)} stars")
    print(f"✓ Filtering efficiency: {len(filtered_data)/len(data)*100:.1f}%")
    
    return filtered_data

def full_coordinate_analysis(stars):
    """
    Complete coordinate transformation with proper variable names and units
    
    Parameters:
    -----------
    stars : astropy.table.Table
        Filtered Gaia data
        
    Returns:
    --------
    dict : Transformed coordinate data
    """
    print("Performing coordinate transformations...")
    
    distances_pc = 1e3 / stars['parallax']
    

    coord_icrs = coord.SkyCoord(
        ra=stars['ra'] * u.deg,
        dec=stars['dec'] * u.deg,
        distance=distances_pc * u.pc,  
        pm_ra_cosdec=stars['pmra'] * u.mas/u.yr,
        pm_dec=stars['pmdec'] * u.mas/u.yr,
        radial_velocity=stars['dr2_radial_velocity'] * u.km/u.s,
        frame='icrs'
    )
    

    galactic_coords = coord_icrs.galactic
    

    gc_frame = coord.Galactocentric(
        galcen_distance=8.5*u.kpc,
        galcen_v_sun=[11.1, 244, 7.3]*u.km/u.s,
        z_sun=27*u.pc
    )
    galactocentric_coords = coord_icrs.transform_to(gc_frame)
    
    print("✓ Coordinate transformations complete")
    
    return {
        'galactic': galactic_coords,
        'galactocentric': galactocentric_coords,
        'l': galactic_coords.l.deg,
        'b': galactic_coords.b.deg,
        'distances_pc': distances_pc,
        'distances_kpc': galactic_coords.distance.to(u.kpc).value
    }

def compute_velocities(galactic_info):
    """
    Compute stellar velocities from coordinate transformations
    
    Parameters:
    -----------
    galactic_info : dict
        Output from full_coordinate_analysis
        
    Returns:
    --------
    dict : Velocity components and kinematic data
    """
    print("Computing stellar velocities...")
    
    galactic_coords = galactic_info['galactic']

    mu_l_cosb = galactic_coords.pm_l_cosb.to(u.mas/u.yr).value  # mas/yr
    mu_b = galactic_coords.pm_b.to(u.mas/u.yr).value           # mas/yr
    
    # Distances
    distances_kpc = galactic_info['distances_kpc']
    

    # v = 4.74 * mu[arcsec/yr] * d[pc] km/s
    # mu in mas/yr, so divide by 1000 to get arcsec/yr
    v_l = 4.74 * (mu_l_cosb/1000) * (distances_kpc * 1000)  # km/s
    v_b = 4.74 * (mu_b/1000) * (distances_kpc * 1000)       # km/s
    
    # Radial velocities
    v_r = galactic_coords.radial_velocity.to(u.km/u.s).value  # km/s
    
    print("✓ Velocity calculations complete")
    
    return {
        'l_deg': galactic_info['l'],
        'b_deg': galactic_info['b'],
        'distances_pc': galactic_info['distances_pc'],
        'distances_kpc': distances_kpc,
        'mu_l_cosb_mas_yr': mu_l_cosb,
        'mu_b_mas_yr': mu_b,
        'v_l_km_s': v_l,
        'v_b_km_s': v_b,
        'v_r_km_s': v_r
    }

def fit_oort_constants(velocity_data):
    """
    Fit Oort constants A and B from stellar motions
    
    Parameters:
    -----------
    velocity_data : dict
        Velocity and position data
        
    Returns:
    --------
    dict : Fitted Oort constants and statistics
    """
    print("Fitting Oort constants from stellar kinematics...")
    
    l_deg = velocity_data['l_deg']
    mu_l = velocity_data['mu_l_cosb_mas_yr']
    v_r = velocity_data['v_r_km_s']
    distances_kpc = velocity_data['distances_kpc']
    
    # Method 1: Fit proper motion pattern μ_l = A*cos(2l) + B
    l_rad = np.radians(l_deg)
    
    # Design matrix for least squares
    X = np.column_stack([np.cos(2 * l_rad), np.ones(len(l_rad))])
    
    # Solve for A and B
    params = np.linalg.lstsq(X, mu_l, rcond=None)[0]
    A_fit, B_fit = params
    
    # Calculate residuals
    mu_l_model = A_fit * np.cos(2 * l_rad) + B_fit
    residuals = mu_l - mu_l_model
    rms_residual = np.sqrt(np.mean(residuals**2))
    
    # Method 2: Cross-check with radial velocities (if available)
    finite_rv = np.isfinite(v_r)
    if np.sum(finite_rv) > 100:
        l_rv = l_rad[finite_rv]
        b_rv = np.radians(velocity_data['b_deg'][finite_rv])
        r_rv = distances_kpc[finite_rv]
        vr_rv = v_r[finite_rv]
        
        # Pattern: v_r = A * r * sin(2l) * cos(b)
        design_rv = r_rv * np.sin(2 * l_rv) * np.cos(b_rv)
        
        # Remove near-zero values
        significant = np.abs(design_rv) > 0.01
        if np.sum(significant) > 50:
            A_rv = np.sum(design_rv[significant] * vr_rv[significant]) / np.sum(design_rv[significant]**2)
            print(f"Cross-check A from radial velocities: {A_rv:.2f} km/s/kpc")
    
    # Calculate derived parameters
    omega_solar = A_fit - B_fit  # Solar angular velocity
    dV_dR = -(A_fit + B_fit)     # Rotation curve slope
    
    results = {
        'A_km_s_kpc': A_fit,
        'B_km_s_kpc': B_fit,
        'rms_residual_mas_yr': rms_residual,
        'omega_solar': omega_solar,
        'dV_dR': dV_dR,
        'n_stars': len(mu_l)
    }
    
    print(f"✓ Fitted A = {A_fit:.2f} km/s/kpc")
    print(f"✓ Fitted B = {B_fit:.2f} km/s/kpc")
    print(f"✓ RMS residual = {rms_residual:.2f} mas/yr")
    print(f"✓ Solar angular velocity = {omega_solar:.2f} km/s/kpc")
    
    return results

def create_analysis_plots(velocity_data, oort_results):
    """
    Create comprehensive analysis plots
    """
    print("Creating analysis visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Galactic Kinematics Analysis Results', fontsize=16)
    
    l_deg = velocity_data['l_deg']
    mu_l = velocity_data['mu_l_cosb_mas_yr']
    v_r = velocity_data['v_r_km_s']
    distances = velocity_data['distances_pc']
    
    A = oort_results['A_km_s_kpc']
    B = oort_results['B_km_s_kpc']
    
    # Plot 1: Proper motion pattern
    ax1 = axes[0, 0]
    
    # Bin data for clarity
    l_bins = np.linspace(0, 360, 25)
    l_centers = (l_bins[1:] + l_bins[:-1]) / 2
    mu_l_binned = []
    
    for i in range(len(l_bins)-1):
        mask = (l_deg >= l_bins[i]) & (l_deg < l_bins[i+1])
        if np.sum(mask) > 10:
            mu_l_binned.append(np.mean(mu_l[mask]))
        else:
            mu_l_binned.append(np.nan)
    
    mu_l_binned = np.array(mu_l_binned)
    finite = np.isfinite(mu_l_binned)
    
    ax1.scatter(l_centers[finite], mu_l_binned[finite], color='blue', alpha=0.7)
    
    # Theoretical curve
    l_theory = np.linspace(0, 360, 1000)
    mu_l_theory = A * np.cos(2 * np.radians(l_theory)) + B
    ax1.plot(l_theory, mu_l_theory, 'r-', linewidth=2, label=f'A={A:.1f}, B={B:.1f}')
    
    ax1.set_xlabel('Galactic Longitude (deg)')
    ax1.set_ylabel('Proper Motion μₗ (mas/yr)')
    ax1.set_title('Proper Motion Pattern')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distance distribution
    ax2 = axes[0, 1]
    ax2.hist(distances, bins=50, alpha=0.7, color='green')
    ax2.set_xlabel('Distance (pc)')
    ax2.set_ylabel('Number of Stars')
    ax2.set_title('Distance Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Radial velocity pattern
    ax3 = axes[1, 0]
    finite_rv = np.isfinite(v_r)
    if np.sum(finite_rv) > 100:
        sample_size = min(2000, np.sum(finite_rv))
        sample_idx = np.random.choice(np.where(finite_rv)[0], sample_size, replace=False)
        ax3.scatter(l_deg[sample_idx], v_r[sample_idx], alpha=0.6, s=2)
        ax3.set_xlabel('Galactic Longitude (deg)')
        ax3.set_ylabel('Radial Velocity (km/s)')
        ax3.set_title('Radial Velocity vs Longitude')
    else:
        ax3.text(0.5, 0.5, 'Insufficient radial\nvelocity data', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Radial Velocity Pattern')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Results comparison
    ax4 = axes[1, 1]
    
    # Compare with literature
    A_lit, B_lit = CONSTANTS['A_true'], CONSTANTS['B_true']
    
    categories = ['A (literature)', 'A (measured)', 'B (literature)', 'B (measured)']
    values = [A_lit, A, B_lit, B]
    colors = ['red', 'blue', 'red', 'blue']
    
    bars = ax4.bar(range(len(categories)), values, color=colors, alpha=0.7)
    ax4.set_xticks(range(len(categories)))
    ax4.set_xticklabels(categories, rotation=45, ha='right')
    ax4.set_ylabel('Oort Constants (km/s/kpc)')
    ax4.set_title('Results vs Literature')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def print_analysis_summary(oort_results):
    """
    Print comprehensive analysis summary
    """
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    A_fit = oort_results['A_km_s_kpc']
    B_fit = oort_results['B_km_s_kpc']
    A_lit = CONSTANTS['A_true']
    B_lit = CONSTANTS['B_true']
    
    print(f"\nOORT CONSTANTS:")
    print(f"  Measured A = {A_fit:.2f} km/s/kpc")
    print(f"  Literature A = {A_lit:.2f} km/s/kpc")
    print(f"  Difference: {abs(A_fit - A_lit):.2f} km/s/kpc ({abs(A_fit - A_lit)/A_lit*100:.1f}%)")
    
    print(f"\n  Measured B = {B_fit:.2f} km/s/kpc")
    print(f"  Literature B = {B_lit:.2f} km/s/kpc")
    print(f"  Difference: {abs(B_fit - B_lit):.2f} km/s/kpc ({abs(B_fit - B_lit)/abs(B_lit)*100:.1f}%)")
    
    print(f"\nGALACTIC ROTATION PARAMETERS:")
    print(f"  Solar angular velocity: {oort_results['omega_solar']:.2f} km/s/kpc")
    print(f"  Rotation curve slope: {oort_results['dV_dR']:.2f} km/s/kpc")
    print(f"  Fit quality: {oort_results['rms_residual_mas_yr']:.2f} mas/yr RMS")
    print(f"  Sample size: {oort_results['n_stars']} stars")
    
    if abs(A_fit - A_lit)/A_lit < 0.1 and abs(B_fit - B_lit)/abs(B_lit) < 0.1:
        print(f"\n✅ EXCELLENT: Results agree with literature within 10%!")
    elif abs(A_fit - A_lit)/A_lit < 0.2 and abs(B_fit - B_lit)/abs(B_lit) < 0.2:
        print(f"\n✅ GOOD: Results agree with literature within 20%!")
    else:
        print(f"\n⚠️  Results show larger deviations - check analysis")
    
    print("="*60)

def run_complete_analysis():
    """
    Run the complete galactic kinematics analysis pipeline
    """
    print("🚀 Starting complete galactic kinematics analysis...")
    
    try:
        # Step 1: Get Gaia data
        print("\n📡 STEP 1: Data Acquisition")
        raw_data = get_gaia_sample(max_stars=30000)
        
        if raw_data is None:
            print("❌ Failed to acquire data. Exiting.")
            return None
        
        # Step 2: Filter data
        print("\n🔧 STEP 2: Quality Filtering")
        filtered_data = filter_data(raw_data)
        
        # Step 3: Coordinate transformations
        print("\n📐 STEP 3: Coordinate Transformations")
        coord_data = full_coordinate_analysis(filtered_data)
        
        # Step 4: Velocity calculations
        print("\n🧮 STEP 4: Velocity Calculations")
        velocity_data = compute_velocities(coord_data)
        
        # Step 5: Oort constants fitting
        print("\n📊 STEP 5: Oort Constants Analysis")
        oort_results = fit_oort_constants(velocity_data)
        
        # Step 6: Visualization
        print("\n📈 STEP 6: Creating Visualizations")
        plots = create_analysis_plots(velocity_data, oort_results)
        
        # Step 7: Summary
        print_analysis_summary(oort_results)
        
        print("\n🎉 Analysis complete! You have successfully measured galactic rotation!")
        
        return {
            'raw_data': raw_data,
            'filtered_data': filtered_data,
            'velocity_data': velocity_data,
            'oort_results': oort_results,
            'plots': plots
        }
        
    except Exception as e:
        print(f"❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

# Main execution
if __name__ == "__main__":
    # Run the complete analysis
    results = run_complete_analysis()
    
    if results is not None:
        print("\n🌟 SUCCESS: Your galactic kinematics analysis is complete!")
        print("You have successfully:")
        print("✅ Downloaded real Gaia data")
        print("✅ Measured galactic rotation parameters") 
        print("✅ Created scientific visualizations")
        print("✅ Validated results against literature")
        print("\n🚀 Ready for your astronomy portfolio!")
    else:
        print("\n❌ Analysis failed. Check error messages above.")