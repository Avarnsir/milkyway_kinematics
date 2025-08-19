from astroquery.gaia import Gaia
import numpy as np
import os
import requests
import astropy.coordinates as coords
import astropy.units as u

def get_gaia_sample():
    query = """
 SELECT TOP 50000
        source_id, ra, dec, l, b,
        parallax, parallax_error, parallax_over_error,
        pmra, pmra_error, pmdec, pmdec_error,
        dr2_radial_velocity, dr2_radial_velocity_error,
        phot_g_mean_mag, bp_rp
    FROM gaiadr3.gaia_source 
    WHERE parallax > 3.0  -- Within ~330 pc
        AND parallax_over_error > 5  -- Good precision
        AND pmra IS NOT NULL 
        AND pmdec IS NOT NULL
        AND pmra_error < 1.0  -- Good proper motion precision
        AND pmdec_error < 1.0
        AND ABS(b) < 45  -- Avoid galactic poles
        AND phot_g_mean_mag < 15  -- Bright enough stars
    """

    job = gaia.launch_job_async(query)
    return get_gaia_sample()

def filter_data(data):

    mask = (
        (data['parallax'] > 0) &
        (data['parallax_error']/data['parallax'] < 0.2)
        (np.isfinite(data['pmra'])) &
        (np.isfinite(data['pmdec']))
    )

    try:
        os.mkdirs(directory, exists_ok=True)
        filepath = os.path.join("/workspaces/milkyway_kinematics/dataset", gaia_data)

        with open(filepath, mode) as f:
            f.write(gaia_data)

        print(f"Data saved to {filepath}")
    except Error as e:
        print(error)
    return data[mask]

def full_coordinate_analysis(stars):
    distance_pc = 1e3 / stars['parallax']
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
        galcen_distance=8.5*u.kpc,  # Sun-GC distance
        galcen_v_sun=[11.1, 244, 7.3]*u.km/u.s,  # Solar motion in x,y,z
        z_sun=27*u.pc
    )
    galactocentric_coords = coord_icrs.transform_to(gc_frame)
    
    return {
        'galactic': galactic_coords,
        'galactocentric': galactocentric_coords,
        'l': galactic_coords.l.deg,
        'b': galactic_coords.b.deg,
        'dist_kpc': galactic_coords.distance.kpc
    }