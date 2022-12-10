import numpy as np
from scipy import interpolate
import os

avail_sims   = ['DMO', 'TNG', 'SIMBA']

cwd_path = os.path.dirname(__file__)

class CvirModel(object):

    def __init__(self, sim):
        '''
        Class that predicts the concentration from a given simulation model.

        The routine "prediction()" interpolates over pre-loaded tables to make 100
        predictions for each Mvir and z, given a set of input simulation parameters.
        The 100 predictions correspond to theoretical variation due to statistical uncertainty.
        '''

        if sim not in avail_sims:

            raise ValueError("Requested sim, %s, is not available. Choose one of "%sim + str(avail_sims))

        self.sim = sim

        self._load_data()

    def _load_data(self):

        #Some arrays to store values
        intercept = []
        slopes    = []
        Mvir_in   = []
        z_in      = []

        redshifts = np.loadtxt(cwd_path + '/../Data/Snapshot_redshift_index.txt')
        scale_fac = 1/(redshifts[:, 1] + 1)

        #Load all necessary quantities at once outside any likelihood function
        for i in range(34):
            model_tmp = np.load(cwd_path + '/../Data/%s/Model_snap%d.npz'%(self.sim, i))
            intercept.append(model_tmp['intercept'])
            slopes.append(model_tmp['slopes'])
            Mvir_in.append(model_tmp['x'])

        #The maximum number of x-values in any redshift
        #this is equal to 37 for our data (z = 0 has most Mvir points)
        max_ind  = np.argmax([slopes[i].shape[1] for i in range(34)])
        max_xnum = slopes[max_ind].shape[1]

        #Load redshifts outside first set because we need to know the max halo counts
        for i in range(34): z_in.append(np.ones(max_xnum)*np.log10(1 + redshifts[i, 1]))

        #loop over all redshifts and postprocess
        for i in range(34):

            #First copy the intercepts at this redshift
            here = intercept[i].copy()

            #How many extra columns do we need to add
            #so the x-values has size 37?
            append_length = max_xnum - here.shape[1]

            #Add extra columns to array and fill with empty np.NaN values
            here = np.append(here, np.zeros_like(intercept[max_ind])[:, :append_length] + np.NaN, axis = 1)

            #Write the new array back into intercept
            intercept[i] = here

            #Do the same for slopes
            here = slopes[i].copy()
            here = np.append(here, np.zeros_like(slopes[max_ind])[:, :append_length, :] + np.NaN, axis = 1)
            slopes[i] = here

            #Do the same for the input Mvir
            here = Mvir_in[i].copy()
            here = np.append(here, np.zeros_like(Mvir_in[max_ind])[:append_length] + np.NaN, axis = 0)
            Mvir_in[i] = here

        #Now convert from list to numpy array
        intercept = np.asarray(intercept)
        slopes    = np.asarray(slopes)
        Mvir_in   = np.asarray(Mvir_in)

        #Create a masked array. This is just a fancier way of
        #keeping track of the missing values in an array.
        #Wherever mask==True means the corresponding entry
        #in the array has a missing value. We filled missing values
        #with NaNs so using np.isnan() will help us find the missing entries
        #in our arrays
        self.intercept = np.ma.MaskedArray(data = intercept, mask = np.isnan(intercept))
        self.slopes    = np.ma.MaskedArray(data = slopes,    mask = np.isnan(slopes))
        self.Mvir_in   = np.ma.MaskedArray(data = Mvir_in,   mask = np.isnan(Mvir_in))
        self.z_in      = np.ma.MaskedArray(data = z_in,      mask = np.isnan(Mvir_in))

        self.input_Mvir_and_z = (Mvir_in[-1], np.log10(scale_fac))


    def predict(self, Mvir, z, Omega_m, sigma_8, SN1, AGN1, SN2, AGN2):

        '''
        Convenience function that interpolates from a precomputed table
        to provide scaling relation parameters of different halo properties
        with halo mass, M200c, at redshifts 0 <= z <= 6.
        The interpolation is done linearly over log(Mvir), and log(a), where
        a = 1/(1 + z) is the scale factor.

        ---------
        Params
        ---------

        Mvir:  (float, int) or (list, numpy array)
            The halo mass. In units of Msun/h

        z: (float, int) or (list, numpy array)
            The redshift of the halos. Can be float, or array of same size as Mvir
            When needed the function will interpolate between available
            data to estimate parameters at the exact input redshift.

        Omega_m: float
            The fraction of energy density in the Universe made up of matter

        sigma_8: float
            The rms of linear density fluctuations smoothed on a scale
            of 8 Mpc/h

        SN1, AGN1, SN2, AGN2: float
            The astrophysical parameters. These are not used if sim = 'DMO'.
            The meaning of these parameters varies between TNG and SIMBA.

        --------
        Output
        --------

        numpy array:

            Array of cvir (linear, not log). The array has dimension (100, Mvir.size),
            where the 100 predictions provide the statistical uncertainty in the prediction.
            If a requested Mvir or z value is outside the interpolation range,
            the corresponding entry in the output will contain np.NaN values.

        '''

        a    = 1/(1 + z)
        Mvir = np.atleast_1d(np.log10(Mvir))

        #If provided redshift is single value, then
        #use that for all halos.
        if isinstance(a, (int, float)):
            a = np.ones_like(Mvir)*a

        #If predictions are from DMO simulation, then
        #we don't use the values of the astro params
        if self.sim == 'DMO':
            theta = [Omega_m, sigma_8]
        else:
            theta = [Omega_m, sigma_8, SN1, AGN1, SN2, AGN2]

        theory =  self.intercept[:, :, :].copy()
        theory += self.Mvir_in[:, None, :]*self.slopes[:, :, :, 0]
        theory += np.sum(np.log10(theta)[None, None, :]*self.slopes[:, :, :, 1:], axis = -1)

        #Swap so that bootstrap is first axis
        theory  = np.swapaxes(theory, 1, 0)

        #Use bounds_error=False. Instead, the output will be np.NaN
        #when we cannot interpolate
        cvir = interpolate.interpn(self.input_Mvir_and_z, theory.data.T,
                                   np.vstack([Mvir, np.log10(a)]).T,
                                   bounds_error = False)

        return 10**cvir.T
