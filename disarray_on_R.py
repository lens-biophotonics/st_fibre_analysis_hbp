''' Compute local and averaged disarray analysis on the numpy marix R. 
R have to be the results of the orientation analysis.'''

# system
import os
import time
import argparse

# general
import numpy as np
from scipy import stats as scipy_stats
from zetastitcher import InputFile

# custom code
from custom_tool_kit import manage_path_argument, create_coord_by_iter, create_slice_coordinate, \
    search_value_in_txt, pad_dimension, write_on_txt, Bcolors
from custom_image_base_tool import normalize, print_info, plot_histogram, plot_map_and_save
from disarray_tools import estimate_local_disarray, save_in_numpy_file, compile_results_strings, \
    Param, Mode, Cell_Ratio_mode, statistics_base, create_R, structure_tensor_analysis_3d, \
    sigma_for_uniform_resolution, downsample_2_zeta_resolution, CONST




# =================================================== MAIN () ===================================================
def main(parser):


    ## Extract input information FROM TERMINAL =======================
    args           = parser.parse_args()
    R_filepath     = manage_path_argument(args.R_path)
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


    # extract filenames and folders
    R_filename     = os.path.basename(R_filepath)
    process_folder = os.path.basename(os.path.dirname(R_filepath))
    base_path      = os.path.dirname(os.path.dirname(R_filepath))
    param_filepath = os.path.join(base_path, process_folder, param_filename)
    stack_prefix   = R_filepath.split('.')[1]

    # create introductory information
    mess_strings = list()
    mess_strings.append(Bcolors.OKBLUE + '\n\n*** Disarray Analysis ***\n' + Bcolors.ENDC)
    mess_strings.append(' > R matrix:           {}'.format(R_filename))
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

    # create dictionary of parameters
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

    mess_strings.append('\n *** Analysis configuration')
    mess_strings.append(' > Pixel size ratio (z / xy) = {0:0.2f}'.format(ps_ratio))
    mess_strings.append(' > Number of selected stack slices for each ROI ({} x {}): {}'.format(
        shape_P[0], shape_P[1], shape_P[2]))
    mess_strings.append(' > Parallelepiped size: ({0},{1},{2}) pixel ='
                        '  [{3:2.2f} {4:2.2f} {5:2.2f}] um'.format(
        shape_P[0], shape_P[1], shape_P[2],
        shape_P[0] * parameters['px_size_xy'],
        shape_P[1] * parameters['px_size_xy'],
        shape_P[2] * parameters['px_size_z']))

    # print to screen
    for s in mess_strings:
        print(s)

    # clear list of strings
    mess_strings.clear()

    # load R array
    R = np.load((R_filepath))


    # --- DISARRAY AND FRACTIONAL ANISOTROPY ESTIMATION -------------------------------------
   
    # estimate local disarrays and fractional anisotropy, write estimated values also inside R
    mtrx_of_disarrays, mtrx_of_local_fa, shape_G, R = estimate_local_disarray(R, parameters,
                                                                              ev_index=2,
                                                                              _verb=_verbose,
                                                                              _verb_deep=_deep_verbose)

    
    # ---  SAVE R TO NUMPY FILE -------------------------------------------------------------

    # retrieve R array prefix
    R_prefix = R_filename.split('.')[0]
    
    # save results to R.npy
    np.save(R_filepath, R)
    mess_strings.append('\n> R matrix saved to: {}'.format(os.path.dirname(R_filepath)))
    mess_strings.append('> with name: {}'.format(R_filename))

    # print to screen
    for l in mess_strings:
        print(l)

    # clear list of strings
    mess_strings.clear()


    # --- SAVE DISARRAY MATRICES TO NUMPY FILE AND COMPILE RESULTS TXT FILE ------------------
    
    # save disarray matrices (computed with arithmetic and weighted means) to numpy file
    disarray_np_filename = dict()
    for mode in [att for att in vars(Mode) if str(att)[0] is not '_']:
        disarray_np_filename[getattr(Mode, mode)] = save_in_numpy_file(
                                                            mtrx_of_disarrays[getattr(Mode, mode)],
                                                            R_prefix, shape_G,
                                                            parameters, base_path, process_folder,
                                                            data_prefix='MatrixDisarray_{}_'.format(mode))

    # save fractional anisotropy to numpy file
    fa_np_filename = save_in_numpy_file(mtrx_of_local_fa, R_prefix, shape_G, parameters,
                                           base_path, process_folder, data_prefix='FA_local_')

    mess_strings.append('\n> Disarray and Fractional Anisotropy matrices saved to:')
    mess_strings.append('> {}'.format(os.path.join(base_path, process_folder)))
    mess_strings.append('with name: \n > {}\n > {}\n > {}\n'.format(
                                                                disarray_np_filename[Mode.ARITH],
                                                                disarray_np_filename[Mode.WEIGHT],
                                                                fa_np_filename))
    mess_strings.append('\n')


    # --- STATISTICAL ANALYSIS, HISTOGRAMS AND SAVINGS --------------------------------------
    
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
            np.savetxt(csv_filepath, values, delimiter=",", fmt = '%f')
            mess_strings.append('> {}'.format(csv_filepath))

    # save histograms
    if _save_hist:
        mess_strings.append('\n> Histogram plots are saved in:')

        # zip matrices, description and filenames
        for (mtrx, lbl, np_fname) in zip([mtrx_of_disarrays[Mode.ARITH], mtrx_of_disarrays[Mode.WEIGHT],
                                   mtrx_of_local_fa],
                                  ['Local Disarray % (Arithmetic mean)', 'Local Disarray % (Weighted mean)',
                                   'Local Fractional Anisotropy'],
                                  [disarray_np_filename[Mode.ARITH], disarray_np_filename[Mode.WEIGHT],
                                   fa_np_filename]):

            # extract only valid values (different of INV = -1)
            values = mtrx[mtrx != CONST.INV]

            # create file path
            hist_fname = '.'.join(np_fname.split('.')[:-1]) + '.tiff'
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
        abs_max     = np.max([mtrx_of_disarrays[Mode.ARITH].max(), mtrx_of_disarrays[Mode.WEIGHT].max()])
        abs_min     = np.min([mtrx_of_disarrays[Mode.ARITH].min(), mtrx_of_disarrays[Mode.WEIGHT].min()])
        dis_norm_A  = 255 * ((mtrx_of_disarrays[Mode.ARITH]  - abs_min) / (abs_max - abs_min))
        dis_norm_W  = 255 * ((mtrx_of_disarrays[Mode.WEIGHT] - abs_min) / (abs_max - abs_min))

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
    my_parser.add_argument('-r', '--R-path', nargs = '+',
                           help = 'absolute path of of R numpy file to read and analyze) ',
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