''' ### EXAMPLE OF EIGENVECTOR ORDERING AND USE ####################

ew = np.array([10, 1, 100])
ev = np.random.randint(low=-10, high=10, size=9, dtype='l').reshape(3, 3)
ev[:, 0] = ev[:, 0] * 10
ev[:, 1] = ev[:, 1] * 1
ev[:, 2] = ev[:, 2] * 100
print(ew)
print(ev)

# eigenvalues sorted in decrescent order, i.e. w0 > w1 > w2
order = np.argsort(ew)[::-1];  
ew = np.copy(ew[order]);        # eigenvalues
ev = np.copy(ev[:, order]);     # eigenvectors
print()
print(ew)
print(ev)

print()
print(ew[0], ' --> ', ev[:, 0])
print(ew[1], ' --> ', ev[:, 1])
print(ew[2], ' --> ', ev[:, 2])

print('... rotation?')
ev_rotated = np.zeros_like(ev)
for axis in range(ev.shape[1]):
    ev_rotated[:, axis] = check_in_upper_semisphere(ev[:, axis])

print(ew[0], ' --> ', ev_rotated[:, 0])
print(ew[1], ' --> ', ev_rotated[:, 1])
print(ew[2], ' --> ', ev_rotated[:, 2])
##################################################################### '''


# system
import os
import time
import argparse

# general
import numpy as np
from   zetastitcher import InputFile

# custom code
from   custom_tool_kit import manage_path_argument, create_coord_by_iter, create_slice_coordinate, \
       search_value_in_txt, pad_dimension, write_on_txt, Bcolors
from   custom_image_base_tool import normalize, print_info, plot_histogram, plot_map_and_save
from   disarray_tools import estimate_local_disarray, save_in_numpy_file, compile_results_strings, \
       Param, Mode, Cell_Ratio_mode, statistics_base, create_R, structure_tensor_analysis_3d, \
       sigma_for_uniform_resolution, downsample_2_zeta_resolution, CONST


def block_analysis(parall, shape_P, parameters, sigma, _verbose): 
    
    ''' FUNCTION DESCRIPTION______________________________________________
    
    - Structure Tensor Analysis of each data block
       
    - PARAMETERS
      parall:     input parallelepiped (data type: uint8 numpy array)
      shape_P:    size of parallelepiped extracted for orientation analysis (data type: numpy array)
      parameters: analysis parameters  (data type: ...)
      sigma:      ...                  (data type: ...)
      _verbose:   verbose flag         (data type: boolean)'''
      
    
    # initialize empty dictionary
    results = {}
    there_is_cell = False
    there_is_info = False

    # check 'mode_ratio' key for selected methods
    if parameters['mode_ratio'] == Cell_Ratio_mode.MEAN:
        cell_ratio = np.mean(parall)
    elif parameters['mode_ratio'] == Cell_Ratio_mode.NON_ZERO_RATIO:
        cell_ratio = np.count_nonzero(parall) / np.prod(shape_P)
    else:
        print(Bcolors.WARNING +
              '** WARNING: parameters[\'mode_ratio\'] is not recognized: all black data blocks are skipped' +
              Bcolors.ENDC)
        cell_ratio = 0

    # check cell_ratio
    if cell_ratio > parameters['threshold_on_cell_ratio']:
        
        # conduct Fiber Orientation Analysis within the current data block
        there_is_cell = True

        # save cell_ratio to R
        results['cell_ratio'] = cell_ratio
        if _verbose:
            print('   cell_ratio :   ', cell_ratio)

        # blurring (for isotropic FWHM) and downsampling (for isotropic pixel size)
        # parall_down data type: float_32
        parall_down = downsample_2_zeta_resolution(parall,
                                                   parameters['px_size_xy'],
                                                   parameters['px_size_z'],
                                                   sigma=sigma)
        
        # Gradient-based 3D Structure Tensor Analysis 
        # - w : descending ordered eigenvalues
        # - v : ordered eigenvectors
        #       the column v[:,i] corresponds to the eigenvector with eigenvalue w[i].
        # - shape_parameters : dictionary with shape parameters
        w, v, shape_parameters = structure_tensor_analysis_3d(parall_down, _rotation = False)

        # check shape_parameters
        if shape_parameters['strenght'] > 0:
            there_is_info = True

            # save ordered eigenvectors
            results['ev'] = v
            results['ew'] = w
            
            # save shape parameters
            for key in shape_parameters.keys():
                results[key] = shape_parameters[key]

        else:
            if _verbose:
                print('Data block rejected ( no info in freq )')
    else:
        if _verbose:
            print('Data block rejected ( no cell )')

    return there_is_cell, there_is_info, results


def iterate_orientation_analysis(volume, R, parameters, shape_R, shape_P, _verbose = False):
    
    ''' FUNCTION DESCRIPTION______________________________________________
       
    - virtually dissect 'volume' in separate data blocks
    - perform the structure tensor analysis implemented in 'block_analysis' 
      within each data block
    - save results to R
    
    - PARAMETERS
      volume:     input stack volume        (data type: numpy array)
      R:          results matrix            (data type: numpy array)
      parameters: analysis parameters       (data type: ...)
      shape_R:    shape of results matrix   (data type: numpy array)
      shape_P:    size of parallelepiped extracted for orientation analysis (data type: numpy array)
      _verbose:   verbose flag              (data type: boolean)
      
    - RETURN: result matrix (R), iteration count (count)'''
    

    # estimate sigma of blurring for isotropic resolution
    sigma_blur = sigma_for_uniform_resolution(FWHM_xy=parameters['fwhm_xy'],
                                              FWHM_z=parameters['fwhm_z'],
                                              px_size_xy=parameters['px_size_xy'])    
    
    perc  = 0   # % progress
    count = 0   # iteration count
    tot   = np.prod(shape_R)
    print(' > Expected iterations: ', tot)

    for z in range(shape_R[2]):

        if _verbose:
            print('\n\n')

        print('{0:0.1f} % - z: {1:3}'.format(perc, z))

        for r in range(shape_R[0]):

            for c in range(shape_R[1]):

                start_coord = create_coord_by_iter(r, c, z, shape_P)
                slice_coord = create_slice_coordinate(start_coord, shape_P)

                perc = 100 * (count / tot)

                if _verbose:
                    print('\n')

                # save init info to R
                R[r, c, z]['id_block'] = count
                R[r, c, z][Param.INIT_COORD] = start_coord

                # extract parallelepiped
                parall = volume[tuple(slice_coord)]

                # check parallelepiped size (add zero_pad if iteration is on the volume border)
                parall = pad_dimension(parall, shape_P)

                # if the whole data block is not black...
                if np.max(parall) != 0:

                    # ...analyze the extracted parallelepiped with block_analysis()
                    there_is_cell, there_is_info, results = block_analysis(
                        parall,
                        shape_P,
                        parameters,
                        sigma_blur,
                        _verbose)

                    # save info to R[r, c, z]
                    if there_is_cell: R[r, c, z]['cell_info']   = True
                    if there_is_info: R[r, c, z]['orient_info'] = True

                    # save results to R
                    if _verbose:
                        print(' saved to R:  ')

                    for key in results.keys():

                        R[r, c, z][key] = results[key]

                        if _verbose:
                            print(' > {} : {}'.format(key, R[r, c, z][key]))

                else:

                    if _verbose:
                        print(' data block rejected')

                count += 1
                
    return R, count


# =================================================== MAIN () ===================================================
def main(parser):

    ### Extract input information FROM TERMINAL =======================
    args           = parser.parse_args()
    source_path    = manage_path_argument(args.source_path)
    param_filename = args.parameters_filename[0]
    
    # preferences
    _verbose      = args.verbose
    _deep_verbose = args.deep_verbose
    _save_csv     = args.csv
    _save_hist    = args.histogram
    _save_maps    = args.maps
    
    if _verbose:
        print(Bcolors.FAIL + ' *** VERBOSE MODE *** ' + Bcolors.ENDC)
    if _deep_verbose:
        print(Bcolors.FAIL + ' *** DEBUGGING MODE *** ' + Bcolors.ENDC)
    ### ===============================================================

    # extract filenames and folder names
    stack_name     = os.path.basename(source_path)
    process_folder = os.path.basename(os.path.dirname(source_path))
    base_path      = os.path.dirname(os.path.dirname(source_path))
    param_filepath = os.path.join(base_path, process_folder, param_filename)
    stack_prefix   = stack_name.split('.')[0]

    # create introductory information
    mess_strings = list()
    mess_strings.append(Bcolors.OKBLUE + '\n\n*** Structure Tensor Orientation Analysis ***\n' + Bcolors.ENDC)
    mess_strings.append(' > Source path:        {}'.format(source_path))
    mess_strings.append(' > Stack name:         {}'.format(stack_name))
    mess_strings.append(' > Process folder:     {}'.format(process_folder))
    mess_strings.append(' > Base path:          {}'.format(base_path))
    mess_strings.append(' > Parameter filename: {}'.format(param_filename))
    mess_strings.append(' > Parameter filepath: {}'.format(param_filepath))
    mess_strings.append('')
    mess_strings.append(' > PREFERENCES:')
    mess_strings.append('   - _verbose       {}'.format(_verbose))
    mess_strings.append('   - _deep_verbose  {}'.format(_deep_verbose))
    mess_strings.append('   - _save_csv      {}'.format(_save_csv))
    mess_strings.append('   - _save_hist     {}'.format(_save_hist))
    mess_strings.append('   - _save_maps     {}'.format(_save_maps))

    # extract parameters
    param_names = ['roi_xy_pix',
                   'px_size_xy', 'px_size_z',
                   'mode_ratio', 'threshold_on_cell_ratio',
                   'local_disarray_xy_side',
                   'local_disarray_z_side',
                   'neighbours_lim',
                   'fwhm_xy','fwhm_z']

    param_values = search_value_in_txt(param_filepath, param_names)

    # create parameter dictionary 
    parameters = {}
    mess_strings.append('\n\n*** Parameters used:')
    mess_strings.append(' > Parameters extracted from {}\n'.format(param_filename))
    for i, p_name in enumerate(param_names):
        parameters[p_name] = float(param_values[i])
        mess_strings.append('> {} - {}'.format(p_name, parameters[p_name]))

    # acquisition system characteristics: ratio of the pixel size along the z and x-y axes
    ps_ratio = parameters['px_size_z'] / parameters['px_size_xy']

    # size of the analyzed block along the z axis
    shape_P = np.array((int(parameters['roi_xy_pix']),
                        int(parameters['roi_xy_pix']),
                        int(parameters['roi_xy_pix'] / ps_ratio))).astype(np.int32)

    mess_strings.append('\n *** Analysis configuration:')
    mess_strings.append(' > Pixel size ratio (z / xy) = {0:0.2f}'.format(ps_ratio))
    mess_strings.append(' > Number of selected stack slices for each ROI ({} x {}): {}'.format(
        shape_P[0], shape_P[1], shape_P[2]))
    mess_strings.append(' > Parallelepiped size: ({0},{1},{2}) pixel ='
                        '  [{3:2.2f} {4:2.2f} {5:2.2f}] um'.format(
        shape_P[0], shape_P[1], shape_P[2],
        shape_P[0] * parameters['px_size_xy'],
        shape_P[1] * parameters['px_size_xy'],
        shape_P[2] * parameters['px_size_z']))

    # create .txt report file
    txt_info_filename = 'Orientations_INFO_' + stack_prefix + '_' \
                   + str(int(parameters['roi_xy_pix'] * parameters['px_size_xy'])) + 'um.txt'
    txt_info_path = os.path.join(os.path.dirname(source_path), txt_info_filename)

    # print analysis report to screen and write into .txt file all introductory information
    write_on_txt(mess_strings, txt_info_path, _print = True, mode = 'w')

    # clear list of strings
    mess_strings.clear()


    # 1 - OPEN STACK ------------------------------------------------------------------------
    
    # extract data (entire volume 'V')
    volume = InputFile(source_path).whole()
    
    # change axes: (r, c, z) -> (z, y, x)
    volume = np.moveaxis(volume, 0, -1)     

    # compute volume size
    shape_V = np.array(volume.shape)
    pixel_for_slice = shape_V[0] * shape_V[1]
    total_voxel_V   = pixel_for_slice * shape_V[2]

    # print volume size information
    mess_strings.append('\n\n*** Loaded volume size')
    mess_strings.append(' > Size of entire volume:  ({}, {}, {})'.format(shape_V[0], shape_V[1], shape_V[2]))
    mess_strings.append(' > Pixels for stack slice:  {}'.format(pixel_for_slice))
    mess_strings.append(' > Total number of voxels:  {}'.format(total_voxel_V))

    # extract list of math information (strings) about the volume.npy variable
    info = print_info(volume, text='\nVolume informations:', _std=False, _return=True)
    mess_strings = mess_strings + info

    # print volume information to screen and add it to the report .txt file
    write_on_txt(mess_strings, txt_info_path, _print = True, mode = 'a')

    # clear list of strings
    mess_strings.clear()

    
    # 2 - LOOP FOR BLOCK EXTRACTION and ANALYSIS -------------------------------------------
    print('\n\n')
    print(Bcolors.OKBLUE + '*** Start Structure Tensor Analysis... ' + Bcolors.ENDC)

    t_start = time.time()

    # create empty result matrix
    R, shape_R = create_R(shape_V, shape_P)

    # conduct analysis on input volume
    R, count = iterate_orientation_analysis(volume, R, parameters, shape_R, shape_P, _verbose)
    mess_strings.append('\n > Orientation analysis complete.')

    # retrieve information about the analyzed data
    block_with_cell  = np.count_nonzero(R[Param.CELL_INFO])
    block_with_info  = np.count_nonzero(R[Param.ORIENT_INFO])
    p_rejec_cell     = 100 * (1 - (block_with_cell / count))
    p_rejec_info_tot = 100 * (1 - (block_with_info / count))
    p_rejec_info     = 100 * (1 - (block_with_info / block_with_cell))

    # get analysis time
    t_process = time.time() - t_start

    # create result strings
    mess_strings.append('\n\n*** Results of Orientation analysis:')
    mess_strings.append(' > Expected iterations: {}'.format(np.prod(shape_R)))
    mess_strings.append(' > Total    iterations: {}'.format(count))
    mess_strings.append(' > Time elapsed:        {0:.3f} s'.format(t_process))
    mess_strings.append('\n > Total blocks analyzed: {}'.format(count))
    mess_strings.append(' > Block with cell: {0}, rejected from total: {1} ({2:0.1f}%)'.format(
        block_with_cell,
        count - block_with_cell,
        p_rejec_cell))
    mess_strings.append(' > Block with gradient information: {}'.format(block_with_info))
    mess_strings.append(' > rejected from total:             {0} ({1:0.1f}%)'.format(count - block_with_info, p_rejec_info_tot))
    mess_strings.append(' > rejected from block with cell:   {0} ({1:0.1f}%)'.format(
        block_with_cell - block_with_info, p_rejec_info))

    mess_strings.append('\n > R matrix created ( shape: ({}, {}, {}) cells (zyx) )'.format(
        R.shape[0], R.shape[1], R.shape[2]))

    # print information to screen and add it to the report .txt file
    write_on_txt(mess_strings, txt_info_path, _print = True, mode = 'a')

    # clear list of strings
    mess_strings.clear()


    # 3 - DISARRAY AND FRACTIONAL ANISOTROPY ESTIMATION -------------------------------------

    # estimate local disarrays and fractional anisotropy, write estimated values also inside R
    mtrx_of_disarrays, mtrx_of_local_fa, shape_G, R = estimate_local_disarray(R, parameters,
                                                                              ev_index=2,
                                                                              _verb=_verbose,
                                                                              _verb_deep=_deep_verbose)


    # 4a - SAVE R ( updated with estimate_local_disarray() ) TO NUMPY FILE ------------------

    # create result matrix (R) filename:
    R_filename = 'R_' + stack_prefix + '_' + str(int(parameters['roi_xy_pix'] * parameters['px_size_xy'])) + 'um.npy'
    R_prefix   = R_filename.split('.')[0]
    R_filepath = os.path.join(base_path, process_folder, R_filename)

    # save results to R.npy
    np.save(R_filepath, R)
    mess_strings.append('\n> R matrix saved to: {}'.format(os.path.dirname(source_path)))
    mess_strings.append('> with name: {}'.format(R_filename))
    mess_strings.append('\n> Information .txt file saved to: {}'.format(os.path.dirname(txt_info_path)))
    mess_strings.append('> with name: {}'.format(txt_info_filename))

    # print information to screen and add it to the report .txt file
    write_on_txt(mess_strings, txt_info_path, _print = True, mode = 'a')

    # clear list of strings
    mess_strings.clear()

    # 4b - SAVE DISARRAY TO NUMPY FILE AND COMPILE RESULTS TXT FILE -------------------------
    
    # save disarray matrices (computed with arithmetic and weighted means) to numpy file
    disarray_np_filename = dict()
    for mode in [att for att in vars(Mode) if str(att)[0] is not '_']:
        disarray_np_filename[getattr(Mode, mode)] = save_in_numpy_file(
                                                            mtrx_of_disarrays[getattr(Mode, mode)],
                                                            R_prefix, shape_G,
                                                            parameters, base_path, process_folder,
                                                            data_prefix = 'MatrixDisarray_{}_'.format(mode))

    # save fractional anisotropy to numpy file
    fa_np_filename = save_in_numpy_file(mtrx_of_local_fa, R_prefix, shape_G, parameters,
                                           base_path, process_folder, data_prefix = 'FA_local_')

    mess_strings.append('\n> Disarray and Fractional Anisotropy matrices saved to:')
    mess_strings.append('> {}'.format(os.path.join(base_path, process_folder)))
    mess_strings.append('with name: \n > {}\n > {}\n > {}\n'.format(
                                                                disarray_np_filename[Mode.ARITH],
                                                                disarray_np_filename[Mode.WEIGHT],
                                                                fa_np_filename))
    mess_strings.append('\n')

  
    # 5 - STATISTICAL ANALYSIS, HISTOGRAMS AND SAVINGS --------------------------------------
    
    # estimate statistics (see class Stat) of disarray and fractional anisotropy matrices
    disarray_ARITM_stats  = statistics_base(mtrx_of_disarrays[Mode.ARITH], invalid_value = CONST.INV)
    disarray_WEIGHT_stats = statistics_base(mtrx_of_disarrays[Mode.WEIGHT],
                                            w=mtrx_of_local_fa,
                                            invalid_value = CONST.INV)

    fa_stats = statistics_base(mtrx_of_local_fa, invalid_value = CONST.INV)

    # compile/append strings of statistical results
    s1 = compile_results_strings(mtrx_of_disarrays[Mode.ARITH], 'Disarray', disarray_ARITM_stats, 'ARITH', '%')
    s2 = compile_results_strings(mtrx_of_disarrays[Mode.WEIGHT], 'Disarray', disarray_WEIGHT_stats, 'WEIGHT', '%')
    s3 = compile_results_strings(mtrx_of_local_fa, 'Fractional Anisotropy', fa_stats)
    disarray_and_fa_results_strings = s1 + ['\n\n\n'] + s2 + ['\n\n\n'] + s3

    # update mess strings
    mess_strings = mess_strings + disarray_and_fa_results_strings

    # create results .txt filename and path
    txt_results_filename = 'results_disarray_by_{}_G({},{},{})_limNeig{}.txt'.format(
        R_prefix,
        int(shape_G[0]), int(shape_G[1]), int(shape_G[2]),
        int(parameters['neighbours_lim']))

    # save to .csv
    if _save_csv:
        mess_strings.append('\n> CSV files saved to:')

        # save disarray and fractional anisotropy matrices to .csv file
        for (mtrx, np_fname) in zip([mtrx_of_disarrays[Mode.ARITH], mtrx_of_disarrays[Mode.WEIGHT], mtrx_of_local_fa],
                        [disarray_np_filename[Mode.ARITH], disarray_np_filename[Mode.WEIGHT], fa_np_filename]):

            # extract only valid values (different from INV = -1)
            values = mtrx[mtrx != CONST.INV]

            # create .csv file path and save data
            csv_filename = np_fname.split('.')[0] + '.csv'
            csv_filepath = os.path.join(base_path, process_folder, csv_filename)
            np.savetxt(csv_filepath, values, delimiter = ",", fmt = '%f')
            mess_strings.append('> {}'.format(csv_filepath))

    # save histograms
    if _save_hist:
        mess_strings.append('\n> Histogram plots saved to:')

        # zip matrices, description and filenames
        for (mtrx, lbl, np_fname) in zip([mtrx_of_disarrays[Mode.ARITH], mtrx_of_disarrays[Mode.WEIGHT],
                                   mtrx_of_local_fa],
                                  ['Local Disarray % (arithmetic mean)', 'Local Disarray % (weighted mean)',
                                   'Local Fractional Anisotropy'],
                                  [disarray_np_filename[Mode.ARITH], disarray_np_filename[Mode.WEIGHT],
                                   fa_np_filename]):

            # extract only valid values (different from INV = -1)
            values = mtrx[mtrx != CONST.INV]

            # create file path
            hist_fname    = '.'.join(np_fname.split('.')[:-1]) + '.tiff'
            hist_filepath = os.path.join(base_path, process_folder, hist_fname)

            # create histograms and save them to image files
            plot_histogram(values, xlabel=lbl, ylabel = 'Sub-volume occurrence', filepath = hist_filepath)
            mess_strings.append('> {}'.format(hist_filepath))

    # save disarray and fa maps
    if _save_maps:
        mess_strings.append('\n> Disarray and Fractional Anisotropy plots saved to:')

        # disarray value normalization:
        #  - in order to preserve the little differences between ARITM and WEIGH disarray matrices,
        #    these are normalized together  
        #  - invalid values are NOT removed for preserving the original matrix (image) shape
        #  - invalid values (if present) are set to the minimum value
        abs_max    = np.max([mtrx_of_disarrays[Mode.ARITH].max(), mtrx_of_disarrays[Mode.WEIGHT].max()])
        abs_min    = np.min([mtrx_of_disarrays[Mode.ARITH].min(), mtrx_of_disarrays[Mode.WEIGHT].min()])
        dis_norm_A = 255 * ((mtrx_of_disarrays[Mode.ARITH]  - abs_min) / (abs_max - abs_min))
        dis_norm_W = 255 * ((mtrx_of_disarrays[Mode.WEIGHT] - abs_min) / (abs_max - abs_min))

        # define destination folder
        dest_folder = os.path.join(base_path, process_folder)

        # create and save data frames (disarray and fractional anisotropy)
        for (mtrx, np_fname) in zip([dis_norm_A, dis_norm_W, mtrx_of_local_fa],
                                    [disarray_np_filename[Mode.ARITH],
                                     disarray_np_filename[Mode.WEIGHT],
                                     fa_np_filename]):

            # plot frames and save them inside a sub_folder (folder_path)
            folder_path = plot_map_and_save(mtrx, np_fname, dest_folder, shape_G, shape_P)
            mess_strings.append('> {}'.format(folder_path))

    # print information to screen and add it to the results .txt file
    txt_results_filepath = os.path.join(base_path, process_folder, txt_results_filename)
    write_on_txt(mess_strings, txt_results_filepath, _print = True, mode = 'w')


# ================================================= END MAIN () ==================================================


if __name__ == '__main__':

    # ============================================ START  BY TERMINAL ============================================
    my_parser = argparse.ArgumentParser(description='Structure Tensor based 3D Orientation Analysis')
    my_parser.add_argument('-s', '--source-path', nargs = '+',
                           help = 'absolute path of tissue sample to analyze (3D .tiff image file or folder of .tiff image files) ',
                           required = True)
    my_parser.add_argument('-p', '--parameters-filename', nargs = '+',
                           help = 'filename of .txt file including the applied parameters (to be placed in the same folder of the analyzed .tiff stack)', 
                           required = True)
    my_parser.add_argument('-v', action = 'store_true', default = False, dest = 'verbose',
                           help = 'print additional information')
    my_parser.add_argument('-d', action = 'store_true', default = False, dest = 'deep_verbose',
                           help = '[debug mode] - print debugging information, e.g. points, values etc.')
    my_parser.add_argument('-c', action = 'store_true', default = True, dest = 'csv',
                           help='save numpy results also as CSV file')
    my_parser.add_argument('-i', action = 'store_true', default = True, dest = 'histogram',
                                                  help='save result histograms to image files')
    my_parser.add_argument('-m', action = 'store_true', default = True, dest = 'maps',
                                                  help='save disarray and FA maps to image files')

    main(my_parser)
    # ===========================================================================================================