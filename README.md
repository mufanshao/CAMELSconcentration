# Baryon Imprints on halo concentration from CAMELS

This repository hosts both the data and corresponding interpolation scripts for the concentration-mass relation of halos in CAMELS simulations (all three of the dark matter only, TNG, and SIMBA variants) from redshifts 0 < z < 6. The properties and their corresponding scaling parameters are computed as described in Shao et. al, in prep. **Citation to this publication is requested if these scaling parameters are used.**

## QUICKSTART

```

from Interpolator import CvirModel
import numpy as np

print("List of available sims is " + str(Interpolator.avail_sims))

Mvir = 10**np.linspace(9, 14.5, 100) #in units of Msun/h

model = CvirModel(sim = 'TNG') #This loads + organizes the model parameters (KLLR slopes/intercepts)
mean  = model.prediction(Mvir, z = 0.45, Omega_m = 0.35, sigma_8 = 0.85, SN1 = 2, AGN1 = 0.8, SN2 = 2, AGN2 = 0.5) #Now make the actual prediction

```

If you find any errors/bugs in the code, please reach out to dhayaa@uchicago.edu
