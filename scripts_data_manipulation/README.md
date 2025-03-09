This folder contains scripts that I use to manipulate the data, e.g. convert nii files or dcm (dicom) files to png.
Please note that these scripts are created to be used with the data in my repository (i.e. saved in a specific location with specific characteristics), so (probably) they will not work immediately in other setups.

In my case, I downloaded the data from the [ADNI website](https://adni.loni.usc.edu/data-samples/adni-data/) and saved them inside a folder called data. 
Then, I create a folder for each dataset (e.g. fMRI, `ADNI_3Yr_3T`, etc). 
Inside of each one of these folders I created folder for each class (e.g. AD for Alzheimerz's Disease, MCI for Mild Cognitive Impairment, CN for Control).
Then in each one of this folder I extract all the data of the specific class downloaded from the ADNI website.
So, for example, the data for MCI patients for the fMRI dataset, in my case, are saved in `data/ADNI_3_fMRI/MCI` (N.b `ADNI_3_fMRI` is the name that I chose for the fMRI dataset, but you could use whatever name you want)
