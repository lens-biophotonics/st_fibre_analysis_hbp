# system
import os
import argparse

# general
import numpy as np
from numpy import linalg as LA
from scipy import stats as scipy_stats

# image
from skimage.filters import gaussian
from skimage import transform
from scipy import ndimage as ndi

# custom code
from custom_image_base_tool import normalize
from custom_tool_kit        import Bcolors, create_coord_by_iter, create_slice_coordinate



# =================================== CLASSES ===================================

# INSERT DESCRIPTION HERE
class CONST:
    INV = -1

# INSERT DESCRIPTION HERE
class Cell_Ratio_mode:
    NON_ZERO_RATIO = 0.0
    MEAN = 1
    
# Structure Tensor Analysis parameters
class Param:
    ID_BLOCK         = 'id_block'                # unique data block identifier
    CELL_INFO        = 'cell_info'               # 1: if data block is to be analyzed; 0: if rejected by cell_threshold
    ORIENT_INFO      = 'orient_info'             # 1: if data block is to be analyzed; 0: if rejected by cell_threshold
    CELL_RATIO       = 'cell_ratio'              # ratio between cell voxels and total voxels in data block
    INIT_COORD       = 'init_coord'              # absolute coordinate of the block[0,0,0] within the total volume to be analyzed
    EW               = 'ew'                      # eigenvalues in descending order
    EV               = 'ev'                      # eigenvectors: column ev[:,i] is the eigenvector with eigenvalue w[i].
    STRENGHT         = 'strenght'                # gradient strength (w1 .=. w2 .=. w3)
    CILINDRICAL_DIM  = 'cilindrical_dim'         # cylindrical shape size (w1 .=. w2 >> w3)
    PLANAR_DIM       = 'planar_dim'              # planar shape size      (w1 >> w2 .=. w3)
    FA               = 'fa'                      # fractional anisotropy  (0-> isotropic, 1-> max anisotropy)
    LOCAL_DISARRAY   = 'local_disarray'          # local_disarray
    LOCAL_DISARRAY_W = 'local_disarray_w'        # local_disarray using fractional anisotropy for weighting versors
    
# Structure Tensor Analysis statistics
class Stat:
    N_VALID_VALUES = 'n_valid_values'
    MIN = 'min'             # min
    MAX = 'max'             # max
    AVG = 'avg'             # mean
    STD = 'std'             # standard deviation  
    SEM = 'sem'             # standard error of the mean
    MEDIAN = 'median'       # median
    MODE   = 'mode'         # mode
    MODALITY = 'modality'   # arithmetic OR weighted mean mode
    
# Selection of averaging mode for local disarray estimation
class Mode:
    ARITH  = 'arithmetic'
    WEIGHT = 'weighted'



# =================================== METHODS ===================================

def downsample_2_zeta_resolution(vol, px_size_xy, px_size_z, sigma):
    
    ''' FUNCTION DESCRIPTION______________________________________________

    - smooth input data along x-y axes so that the PSFxy width becomes equal to the PSFz width
    - downsample data so as to uniform the pixel size (--> isotropic resolution)

    - PARAMETERS
      vol:          input stack volume                    (data type: uint8)
      px_size_xy:   pixel size in um, x and y axes        (data type: float)
      px_size_z:    pixel size in um, z axis              (data type: float)
      sigma:        std of the LowPass Gaussian filter    (data type: float)

    - RETURN: smoothed/downsampled stack            (data type: np.float32)

    '''

    # estimate new isotropic sizes
    resize_ratio = px_size_xy / px_size_z
    resized_dim  = int(resize_ratio * vol.shape[0])
    downsampled  = np.zeros((resized_dim, resized_dim, vol.shape[2]), dtype = np.float32)

    for z in range(vol.shape[2]):
        
        # convolution with Gaussian kernel for smoothing
        blurred = gaussian(image = vol[:, :, z], sigma = sigma, mode = 'reflect')

        # resize and save to new variable
        downsampled[:, :, z] = normalize(
            img=transform.resize(image = blurred, output_shape = (resized_dim, resized_dim), mode = 'reflect'),
            dtype=np.float32)

    return downsampled


def sigma_for_uniform_resolution(FWHM_xy, FWHM_z, px_size_xy):

    ''' FUNCTION DESCRIPTION______________________________________________

    - For obtaining a uniform resolution over the 3 axes, x-y planes are blurred 
      by means of a Gaussian kernel with sigma = sigma_s.

    - sigma_s depends on the x-y and z resolution (FWHM):
      f_xy convoluted by f_s = f_z, where f_s is the Gaussian profile of the 1D PSF;
      thus, sigma_xy**2 + sigma_s**2 = sigma_z**2.

    - This function computes sigma_s (in um) by converting FWHM_xy and FWHM_z values to sigma_xy and sigma_z,
      and computing sigma_s pixel by pixel sixe along x-y.

    - PARAMETERS
      FWHM_xy:  x-y spatial resolution [um]     (data type: float)
      FWHM_z:   z   spatial resolution [um]     (data type: float)
      px_size_xy:   pixel size, x and y axes    (data type: float)

    - RETURN: SD of Gaussian kernel for spatial LP filtering [pixel]    (data type: float)

    '''

    # estimate variance (sigma**2) from FWHM values     [um**2]
    sigma2_xy = FWHM_xy ** 2 / (8 * np.log(2))
    sigma2_z  = FWHM_z ** 2  / (8 * np.log(2))

    # estimate variance (sigma_s**2) of Gaussian kernel [um**2]
    sigma2_s = np.abs(sigma2_z - sigma2_xy)

    # estimate SD of Gaussian Kernel for spatial LP filtering [um]
    sigma_s = np.sqrt(sigma2_s)

    # return SD [pixel]
    return sigma_s / px_size_xy


def turn_in_upper_semisphere(v, axis = 1):

    ''' FUNCTION DESCRIPTION______________________________________________

    - if versor v is in the "axis < 0" semisphere, 
      turn it in the opposite direction

    - PARAMETERS
      v:    input versor    (data type: np.array)
      axis: axis to check   (data type: int)

    - RETURN: reversed versor

    '''
 
    if v[axis] >= 0:
        v_rot = np.copy(v)
    else:
        v_rot = -np.copy(v)
    return v_rot


def structure_tensor_analysis_3d(vol, _rotation = False):

    ''' FUNCTION DESCRIPTION______________________________________________

	- 3D Structure Tensor Analysis inside the volume 'vol'.

    - Structure Tensor Analysis
    |IxIx  IxIy  IxIz|   ->   |mean(IxIx)  mean(IxIy)  mean(IxIz)|
    |IyIx  IyIy  IyIz|   ->   |mean(IyIx)  mean(IyIy)  mean(IyIz)|
    |IzIx  IzIy  IzIz|   ->   |mean(IzIx)  mean(IzIy)  mean(IzIz)|
       
       (3 x 3 x m x m)   ->   (3 x 3)
       where m : ROI side

    - PARAMETERS
      vol:          input stack volume          (data type: numpy array)
      _rotation:    flag for eigenvector rotation
                    in upper semisphere         (data type: boolean)

    - RETURN: shape_parameters, eigenvectors and corresponding eigenvalues
              (the latter are in descending order)

    '''

    # compute gray level gradient along the x, y and z directions
    gx, gy, gz = np.gradient(vol)

    # compute the gradient's second order moment
    Ixx = ndi.gaussian_filter(gx * gx, sigma=1, mode='constant', cval=0)
    Ixy = ndi.gaussian_filter(gx * gy, sigma=1, mode='constant', cval=0)
    Ixz = ndi.gaussian_filter(gx * gz, sigma=1, mode='constant', cval=0)
    # Iyx = Ixy because matrix is simmetric
    Iyy = ndi.gaussian_filter(gy * gy, sigma=1, mode='constant', cval=0)
    Iyz = ndi.gaussian_filter(gy * gz, sigma=1, mode='constant', cval=0)
    # Izx = Ixz because matrix is simmetric
    # Izy = Iyz because matrix is simmetric
    Izz = ndi.gaussian_filter(gz * gz, sigma=1, mode='constant', cval=0)

    # create structure tensor matrix from the mean values of Ixx, Ixy and Iyy
    ST = np.array([[np.mean(Ixx), np.mean(Ixy), np.mean(Ixz)],
                   [np.mean(Ixy), np.mean(Iyy), np.mean(Iyz)],
                   [np.mean(Ixz), np.mean(Iyz), np.mean(Izz)]])

    # eigenvalues and eigenvectors decomposition
    w, v = LA.eig(ST)
   
    # eigenvalues w in descending order, i.e. w0 > w1 > w2 
    # ( column v[:,i] is the eigenvector corresponding to the eigenvalue w[i] )
    order = np.argsort(w)[::-1]  
    w = np.copy(w[order])
    v = np.copy(v[:, order])

    # flip eigenvectors
    if _rotation: 
        ev = np.zeros_like(v)
        for ev_idx in range(v.shape[1]):
            ev[:, ev_idx] = turn_in_upper_semisphere(v[:, ev_idx], axis=1)
    else:
        ev = np.copy(v)

    # define dictionary of tensor shape parameters
    shape_parameters = dict()

    # fractional anisotropy (0: isotropy; 1: max anisotropy)
    shape_parameters['fa'] = np.sqrt(1/2) * (
        np.sqrt((w[0] - w[1]) ** 2 + (w[1] - w[2]) ** 2 + (w[2] - w[0]) ** 2) / np.sqrt(np.sum(w ** 2))
    )

    # get gradient strength (for background segmentation)
    # (w1 .=. w2 .=. w3)
    shape_parameters['strenght'] = np.sqrt(np.sum(w))

    # compute cylindrical shape size    (w1 .=. w2 >> w3)
    shape_parameters['cilindrical_dim'] = (w[1] - w[2]) / (w[1] + w[2])

    # compute planar shape size         (w1 >> w2 .=. w3)
    shape_parameters['planar_dim'] = (w[0] - w[1]) / (w[0] + w[1])
    
    return (w, ev, shape_parameters)


def create_R(shape_V, shape_P):

    ''' FUNCTION DESCRIPTION______________________________________________

    - create results array R

    - PARAMETERS
      shape_V: size of the whole volume                             (data type: numpy array)
      shape_P: spatial resolution of orientation analysis [pixel]   (data type: numpy array)

    - RETURN: empty R results array

    '''

    if any(shape_V > shape_P):        
        shape_R = np.ceil(shape_V / shape_P).astype(np.int)

        # create empty matrix
        total_num_of_cells = np.prod(shape_R)
        R = np.zeros(
            total_num_of_cells,
            dtype=[(Param.ID_BLOCK, np.int64),              # please, refer to the class Param
                   (Param.CELL_INFO, bool),                 
                   (Param.ORIENT_INFO, bool),               
                   (Param.CELL_RATIO, np.float16),          
                   (Param.INIT_COORD, np.int32, (1, 3)),    
                   (Param.EW, np.float32, (1, 3)),          
                   (Param.EV, np.float32, (3, 3)),          
                   (Param.STRENGHT, np.float16),            
                   (Param.CILINDRICAL_DIM, np.float16),     
                   (Param.PLANAR_DIM, np.float16),          
                   (Param.FA, np.float16),                  
                   (Param.LOCAL_DISARRAY, np.float16),      
                   (Param.LOCAL_DISARRAY_W, np.float16)     
                   ]
        ).reshape(shape_R)

        # initialize info mask to False
        R[:, :, :][Param.CELL_INFO]   = False
        R[:, :, :][Param.ORIENT_INFO] = False

        print('Size of result matrix R:', R.shape)

        return R, shape_R

    else:
        raise ValueError(' Volume array size is smaller than the size of fundamental parallelepipeds. \n'
                         ' Check loaded data or modify analysis parameters...')


def create_stats_matrix(shape):

    ''' FUNCTION DESCRIPTION______________________________________________

    - initialize empty array for statistical results

    - PARAMETERS
      shape: size of result matrix R    (data type: numpy array)

    - RETURN: empty statistics array

    '''

    matrix = np.zeros(
        np.prod(shape),
        dtype=[(Stat.N_VALID_VALUES, np.int32),
               (Stat.MIN,  np.float16),
               (Stat.MAX,  np.float16),
               (Stat.AVG,  np.float16),
               (Stat.STD,  np.float16),
               (Stat.SEM,  np.float16),
               (Stat.MODE, np.float16),
               (Stat.MODALITY, np.float16),
               ]
    ).reshape(shape)

    return matrix


def stats_on_structured_data(input, param, w, invalid_value = None, invalid_par = None, _verb = False):

    ''' FUNCTION DESCRIPTION______________________________________________

    - conduct statistical analysis on selected parameter

    - PARAMETERS
      input:         array with structure equal to the results array R                  (data type: numpy array)
      param:         selected parameter                                                 (data type: element of class Param)
      w:             weights                                                            (data type: weights to use in the averaging operation)
      invalid_value: value indicating invalid elements, used to create the valid_mask   (data type: element of class CONST)
      invalid_par:   if passed, invalid_mask is created applying invalid_value
                     to R[invalid_par] instead of R[param]                              (data type: element of class Param)
      _verb:         verbose flag                                                       (data type: boolean)    

    - RETURN:       structure with statistical results

    '''

    if input is not None and param is not None:

        # create boolean mask of valid values (True if valid, False if invalid)
        if invalid_value is not None:
            if invalid_par is not None:
                valid_mask = input[invalid_par] != invalid_value
            else:
                valid_mask = input[param] != invalid_value

        # estimate statistics of selected parameter
        results = statistics_base(input[param], w=w, valid_mask=valid_mask, _verb=_verb)

        return results

    else:
        return None


def statistics_base(x, w = None, valid_mask = None, invalid_value = None, _verb = False):

    ''' FUNCTION DESCRIPTION______________________________________________

    - evaluate statistics of input data (x), applying weights (w) if passed

    - PARAMETERS
      x:             input data                      (data type: numpy array)
      w:             weights                         (data type: numpy array)
      valid_mask:    mask of valid values to analyze (data type: boolean)
      invalid_value: value indicating invalid elements, used to create the valid_mask IF NOT PASSED (data type: element of class CONST)
      _verb:         verbose flag   (data type: boolean)

    - RETURN: statistical results (data type: dictionary)

    '''
   
    if x is not None:

        if w is not None:
            _w = True
            if _verb:
                print(Bcolors.VERB)
                print('* Statistical analysis with weighted average')

        else:
            w = np.ones_like(x) 
            _w = False
            if _verb:
                print('* Statistical analysis with arithmetic average')

        if _verb:
            print('matrix.shape: ', x.shape)
            if _w:
                print('weights.shape: ', w.shape)

        # define results dictionary
        results = dict()

        # if valid_mask is passed (or created based on invalid_value), only valid values are considered
        if valid_mask is None:
            if invalid_value is None:
                valid_mask = np.ones_like(x).astype(np.bool)
            else:
                valid_mask = (x != invalid_value)

        valid_values  = x[valid_mask]  
        valid_weights = w[valid_mask]

        if _verb:
            print('Number of selected values: {}'.format(valid_values.shape))

        # get number of valid instances, and their min and max values
        results[Stat.N_VALID_VALUES] = valid_values.shape[0]  # [0] because valid_values is a tuple
        results[Stat.MIN]            = valid_values.min()
        results[Stat.MAX]            = valid_values.max()

        # compute data mean, std, SEM, mode and median
        results[Stat.AVG] = np.average(valid_values, axis=0, weights=valid_weights)
        results[Stat.STD] = np.sqrt(
            np.average((valid_values - results[Stat.AVG]) ** 2, axis=0, weights=valid_weights))
        results[Stat.SEM]    = scipy_stats.sem(valid_values)
        results[Stat.MEDIAN] = np.median(valid_values)

        # mode return an object with 2 attributes: mode and count, both as n-array, with n the iniput axis
        results[Stat.MODE] = scipy_stats.mode(valid_values).mode[0]

        # save averaging mode
        results[Stat.MODALITY] = Mode.ARITH if w is None else Mode.WEIGHT

        # print to screen
        if _verb:
            print('Modality:, ', results[Stat.MODALITY])
            print(' - avg:, ', results[Stat.AVG])
            print(' - std:, ', results[Stat.STD])
            print(Bcolors.ENDC)

        return results

    else:
        if _verb:
            print('ERROR: ' + statistics_base.__main__ + ' - Matrix passed is None')

        return None


def estimate_local_disarray(R, parameters, ev_index = 2, _verb = True, _verb_deep = False):

    ''' FUNCTION DESCRIPTION______________________________________________

    - given R and:
      . according with the space resolution defined in 'parameters'
      . using only the eigenvectors with index 'ev_index'
      
      estimate the local_disarray (in the versors cluster) over the whole input R matrix.
    
      Local disarray is defined as: 100 * (1 - align),
      where local align is estimated by the module of the average vector of the local versors.

      Two different averaging are applied:
      . arithmetic average (version 0);
      . weighted average, where every vector is weighted 
        by the corresponding value of Fractional Anisotropy (version 1).

      Both versions are saved to the structured variable 'matrices_of_disarrays', 
      as:
      . matrices_of_disarrays['arithmetic'] 
      . matrices_of_disarrays['weighted']

    - PARAMETERS
      R:            results matrix          (data type: numpy structured array )
      parameters:   dictonary of parameters (data type: dict)
      ev_index:     eigenvector index       (data type: int [0, 1 or 2])
      _verb:        verbose flag            (data type: boolean)
      _verb_deep:   deep verbose flag       (data type: boolean)

    - RETURN: results array R, disarray and fractional anisotropy matrices, shape of disarray analysis grane

    '''

    # import spatial resolution
    res_xy = parameters['px_size_xy']
    res_z  = parameters['px_size_z']
    resolution_factor = res_z / res_xy

    # estimate the spatial resolution of R
    block_side      = int(parameters['roi_xy_pix'])
    num_of_slices_P = block_side / resolution_factor
    shape_P         = np.array((block_side, block_side, num_of_slices_P)).astype(np.int32)

    # print to screen
    if _verb:
        print('\n\n*** Estimate_local_disarray()')
        print('R settings:')
        print('> R.shape:       ', R.shape)
        print('> R[ev].shape:   ', R['ev'].shape)
        print('> Pixel size ratio (z / xy) = ', resolution_factor)
        print('> Number of slices selected in R for each ROI ({} x {}): {}'.format(block_side, block_side, num_of_slices_P))
        print('> Size of parallelepiped in R:', shape_P, 'pixel = [{0:2.2f} {0:2.2f} {1:2.2f}] um'.format(
            block_side * res_xy, num_of_slices_P * res_z))

    # import disarray space resolution
    Ng_z  = parameters['local_disarray_z_side']
    Ng_xy = parameters['local_disarray_xy_side']
    neighbours_lim = parameters['neighbours_lim']

    # check the validity of the disarray space resolution in the image plane
    if Ng_xy == 0:
        Ng_xy = int(Ng_z * resolution_factor)
    elif Ng_xy < 2:
        Ng_xy = 2
    
    # print to screen
    if _verb:
        print('Disarray settings:')
        print('> Grane size in XY: ', Ng_xy)
        print('> Grane size in Z:  ', Ng_z)
        print('> neighbours_lim for disarray: ', neighbours_lim)

    # shape of the disarray analysis 'grane' (sub-volume)
    shape_G = (int(Ng_xy), int(Ng_xy), int(Ng_z))

    # estimate expected iterations along each axis
    iterations = tuple(np.ceil(np.array(R.shape) / np.array(shape_G)).astype(np.uint32))
    if _verb:
        print('\n\n> Expected iterations on each axis: ', iterations)
        print('\n\n> Expected total iterations       : ', np.prod(iterations))

    # define matrices including local disarray and local FA data
    # - local disarray: disarray of selected cluster of orientation versors
    # - local FA:       mean of the FA of the selected cluster of orientation versors
    matrices_of_disarray              = dict()
    matrices_of_disarray[Mode.ARITH]  = np.zeros(iterations).astype(np.float32)
    matrices_of_disarray[Mode.WEIGHT] = np.zeros(iterations).astype(np.float32)
    matrix_of_local_fa                = np.zeros(iterations).astype(np.float32)

    # print to screen
    if _verb: print('\n *** Start elaboration...')

    # open colored session
    if _verb_deep: print(Bcolors.VERB)  
    _i = 0
    for z in range(iterations[2]):
        for r in range(iterations[0]):
            for c in range(iterations[1]):
                if _verb_deep:
                    print(Bcolors.FAIL + ' *** DEBUGGING MODE ACTIVATED ***')
                    print('\n\n\n\n')
                    print(Bcolors.WARNING +
                          'iter: {0:3.0f} - (z, r, c): ({1}, {2} , {3})'.format(_i, z, r, c) +
                          Bcolors.VERB)
                else:
                    print('iter: {0:3.0f} - (z, r, c): ({1}, {2} , {3})'.format(_i, z, r, c))

                # extract sub-volume ('grane') with size (Gy, Gx, Gz) from R matrix
                start_coord = create_coord_by_iter(r, c, z, shape_G)
                slice_coord = tuple(create_slice_coordinate(start_coord, shape_G))
                grane = R[slice_coord]   
                if _verb_deep:
                    print(' 0 - grane -> ', end = '')
                    print(grane.shape)

                # N = Gy*Gx*Gz = n° of orientation blocks
                # (gx, gy, gz) --> (N,)
                grane_reshaped = np.reshape(grane, np.prod(grane.shape))
                if _verb_deep:
                    print(' 1 - grane_reshaped --> ', end = '')
                    print(grane_reshaped.shape)

                n_valid_cells = np.count_nonzero(grane_reshaped[Param.CELL_INFO])
                if _verb_deep:
                    print(' valid_cells --> ', n_valid_cells)
                    print(' valid rows  --> ', grane_reshaped[Param.CELL_INFO])
                    print(' grane_reshaped[\'cell_info\'].shape:', grane_reshaped[Param.CELL_INFO].shape)

                if n_valid_cells > parameters['neighbours_lim']:

                    # (N) -> (N x 3) (select the eigenvector with index 'ev_index' from the N cells available)
                    coord = grane_reshaped[Param.EV][:, :, ev_index]
                    if _verb_deep:
                        print(' 2 - coord --> ', coord.shape)
                        print(coord)

                    # extract fractional anisotropy
                    fa = grane_reshaped[Param.FA]

                    # print lin.norm and FA components of every versors (iv = index_of_versor)
                    if _verb_deep:
                        for iv in range(coord.shape[0]):
                            print(iv, ':', coord[iv, :],
                                  ' --> norm: ', np.linalg.norm(coord[iv, :]),
                                  ' --> FA:   ', fa[iv])

                    # select only versors and FAs estimated from valid cells:
                    valid_coords = coord[grane_reshaped[Param.CELL_INFO]]
                    valid_fa     = fa[grane_reshaped[Param.CELL_INFO]]
                    
                    # print to screen
                    if _verb_deep:
                        print(' valid coords - ', valid_coords.shape, ' :')
                        print(valid_coords)
                        print(' valid FAs - ', valid_fa.shape, ' :')
                        print(valid_fa)

                    # order valid versors by their FA ('-fa' is applied for descending order)
                    ord_coords = valid_coords[np.argsort(-valid_fa)]
                    ord_fa     = valid_fa[np.argsort(-valid_fa)]

                    # take first versor (with highest FA) for moving all versors in the same half-space
                    v1 = ord_coords[0, :]

                    # move all versors in the same congruent direction
                    # (by checking the sign of their dot product between the first one, v1)
                    if _verb_deep: print('Check if versors have congruent direction (i.e. lie in the same half-space)')
                    
                    for iv in range(1, ord_coords.shape[0]):

                        scalar = np.dot(v1, ord_coords[iv])

                        if scalar < 0:
                            # change direction of the i-th versor
                            if _verb_deep: print(ord_coords[iv], ' --->>', -ord_coords[iv])
                            ord_coords[iv] = -ord_coords[iv]

                    # print definitive lin.norm and FA components of each versor
                    if _verb_deep:
                        print('Definitive versor components in the same half-space:')
                        for iv in range(ord_coords.shape[0]):
                            print(iv, ':', ord_coords[iv, :],
                                  ' --> norm: ', np.linalg.norm(ord_coords[iv, :]),
                                  ' --> FA:   ', ord_fa[iv])

                    if _verb_deep:
                        print('np.average(ord_coords): \n', np.average(ord_coords, axis = 0))
                        print('np.average(ord_coords, weight = fa): \n',
                              np.average(ord_coords, axis = 0, weights = ord_fa))

                    # compute the alignment degree as the modulus of the average vector
                    # (arithmetic and weighted wrt to the local FA)
                    alignment = dict()
                    alignment[Mode.ARITH]  = np.linalg.norm(np.average(ord_coords, axis=0))
                    alignment[Mode.WEIGHT] = np.linalg.norm(np.average(ord_coords, axis=0, weights=ord_fa))
                    if _verb_deep:
                        print('alignment[Mode.ARITH] : ', alignment[Mode.ARITH])
                        print('alignment[Mode.WEIGHT]: ', alignment[Mode.WEIGHT])

                    # compute the local_disarray degree
                    local_disarray = dict()
                    local_disarray[Mode.ARITH]  = 100 * (1 - alignment[Mode.ARITH])
                    local_disarray[Mode.WEIGHT] = 100 * (1 - alignment[Mode.WEIGHT])

                    # save the disarray estimated from each grane of the
                    # for plots and statistical analysis
                    R[slice_coord][Param.LOCAL_DISARRAY]   = local_disarray[Mode.ARITH]
                    R[slice_coord][Param.LOCAL_DISARRAY_W] = local_disarray[Mode.WEIGHT]

                    # estimate the average Fractional Anisotropy
                    # and save results to local disarray matrices
                    matrix_of_local_fa[r, c, z] = np.mean(ord_fa)
                    matrices_of_disarray[Mode.ARITH][r, c, z]  = local_disarray[Mode.ARITH]
                    matrices_of_disarray[Mode.WEIGHT][r, c, z] = local_disarray[Mode.WEIGHT]

                    # print to screen
                    if _verb_deep:
                        print('saving... rcz:({},{},{}):'.format(r, c, z))
                        print('local_disarray[Mode.ARITH] : ', local_disarray[Mode.ARITH])
                        print('local_disarray[Mode.WEIGHT]: ', local_disarray[Mode.WEIGHT])
                        print('mean Fractional Anisotropy : ', matrix_of_local_fa[r, c, z])

                else:
                    # assign invalid value (-1)
                    # (assume that isolated quiver has no disarray)
                    R[slice_coord][Param.LOCAL_DISARRAY]   = -1
                    R[slice_coord][Param.LOCAL_DISARRAY_W] = -1
                    matrices_of_disarray[Mode.ARITH][r, c, z]  = -1
                    matrices_of_disarray[Mode.WEIGHT][r, c, z] = -1
                    matrix_of_local_fa[r, c, z] = -1

                # end iteration
                _i += 1

    # close colored session
    print(Bcolors.ENDC)

    return matrices_of_disarray, matrix_of_local_fa, shape_G, R


def save_in_numpy_file(array, R_prefix, shape_G, parameters,
                       base_path, process_folder, data_prefix=''):

    ''' FUNCTION DESCRIPTION______________________________________________

    - save disarray matrix to numpy file

    - PARAMETERS
      array:    input disarray data                         (data type: numpy array)
      R_prefix: prefix of output R filename                 (data type: string)
      shape_G:  local disarray spatial resolution (pixel)   (data type: numpy array)
      parameters:   dictonary of parameters                 (data type: dict)
      base_path:  base folder path                          (data type: string)
      process_folder: last folser path                      (data type: string)
      data_prefix: string to add to the output name         (data type: string)

    - RETURN: filename  (data type: string)

    '''

    numpy_filename_endname = '{}_G({},{},{})_limNeig{}.npy'.format(
        R_prefix,
        int(shape_G[0]), int(shape_G[1]), int(shape_G[2]),
        int(parameters['neighbours_lim']))

    disarray_numpy_filename = data_prefix + numpy_filename_endname
    np.save(os.path.join(base_path, process_folder, disarray_numpy_filename), array)

    return disarray_numpy_filename


def compile_results_strings(matrix, name, stats, mode='none_passed', ext=''):

    ''' FUNCTION DESCRIPTION______________________________________________

    - compile result strings

    - PARAMETERS
      matrix:   disarray matrix evaluated  (data type: np.array)
      name:     parameter name             (data type: string)    
      stats:    statistical results        (data type: ...)
      mode:     averaging mode, either 'aritm' or 'weighted'    (data type: string)
      ext:      results extension, e.g. % um None               (data type: string)

    - RETURN:   strings (data type: string list)

    '''

    strings = list()

    strings.append('\n\n *** Statistical analysis of {} on accepted points. \n'.format(name))
    strings.append('> {} matrix with shape:  {}'.format(name, matrix.shape))
    strings.append('> Number of valid values:   {}'.format(stats[Stat.N_VALID_VALUES]))
    strings.append('> Statistical evaluation modality: {}\n'.format(mode))
    strings.append('> Statistical results:')

    # generate string for each statistical parameter
    for att in [att for att in vars(Stat) if str(att)[0] is not '_']:
  
        # check if parameter includes a string (e.g. 'ARITH' or 'WEIGHT')
        if isinstance(stats[getattr(Stat, att)], str):
            strings.append(' - {0}: {1}'.format(getattr(Stat, att), stats[getattr(Stat, att)]))

        elif att == Stat.N_VALID_VALUES:
            # print integer, without extension
            strings.append(' - {0}: {1}'.format(getattr(Stat, att), stats[getattr(Stat, att)]))
        else:
            strings.append(' - {0}: {1:0.2f}{2}'.format(getattr(Stat, att), stats[getattr(Stat, att)], ext))

    return strings

