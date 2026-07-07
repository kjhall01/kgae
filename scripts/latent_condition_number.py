

import numpy as np
import xarray as xr



if __name__ == "__main__":
    import kgae
    import numpy as np 

    experiment = 'large-ensemble-1940-2014'
    n_ensemble = 201
    ci = 0.9
    modes_to_examine = [5,4,3]
    n_modes = len(modes_to_examine)


    # ---- usage ----
    # Z: xr.DataArray with dims ('seed','mode','time')  (and optionally others)
    cond_ZTZ = cond_ztz_per_point(Z)


    # Alternative: condition number of the design matrix X (time x mode).
    cond_X = np.sqrt(cond_ZTZ).rename("cond_X") 
