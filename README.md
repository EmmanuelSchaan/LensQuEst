#  CMB lensing: standard quadratic estimator, shear and magnification (https://arxiv.org/abs/1804.06403).


Manipulate flat sky maps (FFT, filtering, power spectrum, generating Gaussian random field, applying lensing to a map, etc).
Forecast the noise on CMB lensing estimators (standard, shear-only, magnification-only).
Evaluate these estimators on flat sky maps.

To get setup with required packages
```
conda env create -f environment.yml
conda activate nblensing
python -m ipykernel install --user --name nblensing --display-name "nblensing"
```

Demo in `demos/demo.ipynb`

Hope you find this code useful! Please cite https://arxiv.org/abs/1804.06403 if you use this code in a publication. Do not hesitate to contact me with any questions: eschaan@lbl.gov
