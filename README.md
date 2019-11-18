# pySearchlight
python tool for fMRI searchlight mapping

# to install : 
python setup.py develop

# to run tests :
python setup.py test

# example case

```python
import nibabel as nib
import numpy as np
from searchlight.searchlight import RSAsearchlight

# we will perform a searchlight with a radius of 3
radius = 3

# get a binary mask of the subject's brain
mask = nib.load('mask.nii').get_data()

# get a x y z by conditions volume of fmri patterns
betas = nib.load('betas.nii').get_data()

assert mask.shape==betas.shape[0:2], 'dimensions of mask and betas must match.'

# initialise the searchlight.
SL = RSAsearchlight(
        mask, # pass the binary mask
        radius=3, # radius of 3
        thr=1.0, # threshold of 1
        njobs=2, # this will distribute the searchlight mapping on 2 cores.
        verbose=True  # this will make use of tqdm to display time spent and left
        )

SL.fit_rsa(
    betas, # pass in the betas
    wantreshape=True # the resulting SL.RDM is reshapes in x, y, z, n_pairs
    )
```

# planned enhancements:

- this searchlight package will ultimately sit on pyrsa.
- e.g. one could import a distance metric from pyrsa.rdm.calc and pass it to SL.fit_rsa as a fitting method.
