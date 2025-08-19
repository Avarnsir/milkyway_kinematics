from astroquery.gaia import Gaia
import numpy as np
import os
import requests

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