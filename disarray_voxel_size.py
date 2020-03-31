import numpy as np


def main():
	
    ''' SCRIPT DESCRIPTION______________________________________________
    
    - extract pixel size information in the averaged volumes 
      (e.g. disarray and local_FA matrices)

    '''

	# =============== INPUT PARAMETERS  ===============

	# volume size in pixel
    shape_V = np.array([3192, 2316, 1578])  

	# resolution adopted in the analysis of the R matrix [um]
    res_array_rcz = np.array([5.2, 5.2, 6])

	# size of the block analysis
    shape_P = np.array([26,26,22])

	# disarray matrix size
    shape_D = np.array([62,45,36])

	# size of the grane analysis of disarray matrices
    shape_G = np.array([2,2,2]).astype(np.uint32)

	# Insert here the plot disarray block sizes [pixel]
	# (to be manually measured, e.g. with FIJI)
    r_dis = 19
    c_dis = 19

	# =================================================
    
    # retrieve x-y and z resolutions
    res_xy = res_array_rcz[1]
    res_z  = res_array_rcz[2]

    # resolution of disarray analysis [um]
    disarray_grane_um = res_array_rcz * shape_V / shape_D

    # print to screen
    print('Vol: ', shape_V, '\n - disarray matrix: ', shape_D)
    print('--> disarray estimation grane [px]: ',shape_P * shape_G)
    print('--> disarray estimation grane [um]: ',disarray_grane_um)

    print('\n3D resolution of the disarray plot (to be set in FIJI):')
    print('r, c -> ({0:0.2f}, {1:0.2f}) um = (res_xy * roi_xy_pix * shape_G / block_side_in_plot)'.format(
	    res_xy * shape_P[0] * shape_G[0] / r_dis,
	    res_xy * shape_P[1] * shape_G[1] / c_dis))
    print('z  ---> {0:0.2f} um'.format(disarray_grane_um[2]))

    # check values
    print('\n\n')
    print("Size proof:")
    print('shape_V * res_array_rcz = vol_in_um')
    print(shape_V, '*', res_array_rcz, '=', shape_V * res_array_rcz, 'um')
    print()
    print('shape_D * disarray_grane_um = vol_in_um')
    print(shape_D, '*', disarray_grane_um, '=', shape_D * disarray_grane_um, 'um')
    print()
    print('shape_G * shape_D * shape_P = vol_enlarged_>in_px = vol_enlarged_in_um')
    print(shape_G, '*', shape_D, '*', shape_P, '=', 
	        shape_G * shape_D * shape_P, 'px = ',
	        shape_G * shape_D * shape_P * res_array_rcz, 'um')


if __name__ == '__main__':
    main()