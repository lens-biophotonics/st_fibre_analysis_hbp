# 3D Structure Tensor Analysis
This repository includes a Structure Tensor Analysis (STA) tool for the characterization of local 3D orientation in TIFF image stacks, based on the evaluation of local image intensity gradients.  The present tool also provides a full analysis of local gradient strength, structure disarray, shape and fractional anisotropy indices.

## Python 3 Script
**st_analysis.py**  
*saves a NumPy structured array including the information below for each (valid) data block in the input TIFF stack*

* `EW`: structure tensor eigenvalues *w* (in descending order)
* `EV`: structure tensor eigenvectors *v*
* `STRENGTH`: gradient strength (*w*1 .=. *w*2 .=. *w*3)
* `CILINDRICAL_DIM`: cylindrical shape size (*w*1 .=. *w*2 >> *w*3)
* `PLANAR_DIM`: planar shape size (*w*1 >> *w*2 .=. *w*3)
* `FA`: fractional anisotropy index
* `LOCAL_DISARRAY`: local disarray index
* `LOCAL_DISARRAY_W`: local disarray index weighted by fractional anisotropy

*(optional) saves local disarray and fractional anisotropy data to .csv file*

*(optional) saves data histograms to image files*

*(optional) saves disarray and fractional anisotropy maps to image files*

## Python Modules
**custom_image_base_tool.py**  
**custom_tool_kit.py**  
**disarray_tools.py**

---
## Configuration
The parameter values of the STA algorithm must be assigned through a .txt configuration file stored within the input image file directory.   
They are listed below with their definition:

##### Optical System
* `px_size_xy`: in-plane pixel size [μm]
* `px_size_z`:  through-plane pixel size [μm]
* `fwhm_xy`: in-plane FWHM of the optical system PSF [μm]
* `fwhm_z`: through-plane FWHM of the optical system PSF [μm]

##### STA Resolution
* `roi_xy_pix`: in-plane STA data block size [pixel]  
    (`roi_z_pix = roi_xy_pix * px_size_xy / px_size_z` is applied for isotropic resolution )
       
##### Data Selection
* `mode_ratio`: flag defining the thresholding strategy for pre-selecting the data blocks fed to the algorithm
    * `0` -> 'non_zero_ratio', i.e. the minimum ratio of non-zero pixels allowed in the data block
    * `1` -> 'mean', i.e. the minimum mean graylevel allowed in the data block
* `threshold_on_cell_ratio`: threshold value applied to each data block before the STA is run  
(e.g. `0.75` for 'non_zero_ratio', or `10` for 'mean')
##### Local Disarray Analysis
* `local_disarray_xy_side`: in-plane mask size for analyzing the local disarray in the resulting vector maps [pixel]
* `local_disarray_z_side`: through-plane local disarray mask size [pixel]
* `neighbours_lim`: minimum number of valid neighbouring data blocks enabling the local disarray evaluation

---
## Example
To execute the program from the command line run:
```python
python3 st_analysis.py -s stack_name.tif -p sta_param.txt
``` 
Other input arguments not shown in the present example can be accessed via the script `--help` option.

As in the example above, the `-p` keyword must be used to assign the algorithm configuration by means of the text file `sta_param.txt`. 
A sample content of this file is reported below:
```Python
px_size_xy = 0.439              # [um]
px_size_z  = 1                  # [um]
fwhm_xy    = 0.731              # [um]
fwhm_z     = 3.09               # [um]
roi_xy_pix = 92                 # [pixel]
mode_ratio = 0                  # 'non_zero_ratio' thresholding mode
threshold_on_cell_ratio = 0.85  # 85% non-zero pixel threshold
```
