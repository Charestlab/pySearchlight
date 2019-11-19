# pySearchlight
python tool for fMRI searchlight mapping

# to install : 
`python setup.py develop`

# to run tests :
`python setup.py test`

# example case

```python
import nibabel as nib
import numpy as np
from searchlight.searchlight import RSAsearchlight
from searchlight.utils import corr_rdms
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

# we will perform a searchlight with a radius of 3
radius = 3

# get a binary mask of the subject's brain
mask = nib.load('mask.nii').get_data()

# get a x y z by conditions volume of fmri patterns
betas = nib.load('betas.nii').get_data()
x, y, z, n_conditions = betas.shape

assert mask.shape==betas.shape[0:3], 'dimensions of mask and betas must match.'

# initialise the searchlight.
SL = RSASearchLight(
        mask, # pass the binary mask
        radius=3, # radius of 3
        threshold=1.0, # threshold of 1
        njobs=2, # this will distribute the searchlight mapping on 2 cores.
        verbose=True  # this will make use of tqdm to display time spent and left
        )

SL.fit_rsa(
    betas, # pass in the betas
    wantreshape=False # the resulting SL.RDM is shape n_centers, n_pairs
    )

# SL.RDM is a numpy array with n_centers, n_comps (upper triangular vector)
# SL.RDM (if wantreshape was set to True when we fit, would be a x , y , z by n_pairs array)

index = 100
plt.imshow(squareform(SL.RDM[index, :])) # plots the RDM of the 100th center 

# let's make model inference
n_pairs = n_conditions*(n_conditions-1) // 2 
model_rdm = np.random.rand(1, n_pairs)

rdm_corr_to_model = corr_rdms(SL.RDM, model_rdm)

brain_vol = np.zeros((x, y, z))
brain_vol[SL.centerIndices] = rdm_corr_to_model
# then you can use matplotlib imshow with searchlight.utils.makeimagestack(brain_vol) 
# for a quick visualisation.

```

# planned enhancements:

- this searchlight package will ultimately sit on pyrsa.
- e.g. one could import a distance metric from pyrsa.rdm.calc and pass it to SL.fit_rsa as a fitting method.
- write an nibabel backed write_output module.