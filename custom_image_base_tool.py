import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io  import BytesIO
from skimage.external.tifffile import imsave
from custom_tool_kit import magnitude


class ImgFrmt:
    # image_format STRINGS
    EPS  = 'EPS'
    TIFF = 'TIFF'
    SVG  = 'SVG'


def plot_map_and_save(matrix, np_filename, base_path, shape_G, shape_P, img_format = ImgFrmt.TIFF, _do_norm = False):
    
    ''' FUNCTION DESCRIPTION______________________________________________
    
    - plot LOCAL disarray or averaged Fractional Anisotropy matrices as frames

     example 1: plot_map_and_save(matrix_of_disarray, disarray_numpy_filename, True, IMG_TIFF)
     example 2: plot_map_and_save(matrix_of_local_FA, FA_numpy_filename, True, IMG_TIFF)
    
    - PARAMETERS
      matrix:       disarray or fractional anisotropy data                        (data type: numpy array)
      np_filename:  filename of results numpy file (to define output filename)    (data type: string)
      shape_G:      local disarray spatial resolution (pixel)                     (data type: numpy array)
      shape_P:      fiber orientation spatial resolution (pixel)                  (data type: numpy array) 
      img_format:   output image format                                           (data type: string) 
      _do_norm:     data normalization flag                                       (data type: bool) 
      
      map_name = 'FA', or 'DISARRAY_ARIT' or 'DISARRAY_WEIGH' '''
    

    # create folder path and filename from np_filename
    plot_folder_name  = np_filename.split('.')[0]
    plot_filebasename = '_'.join(np_filename.split('.')[0].split('_')[0:2])

    # create path where to save the output images
    plot_path = os.path.join(base_path, plot_folder_name)

    # check if already present
    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)

    # iteration along the z axis
    for i in range(0, matrix.shape[2]):

        # extract data to plot
        if _do_norm:
            img = normalize(matrix[..., i])
        else:
            img = matrix[..., i]

        # get depth in the volume space
        z_frame = int((i + 0.5) * shape_G[2] * shape_P[2])

        # define figure title
        title = plot_filebasename + '. Grane: ({} x {} x {}) vectors; Depth_in_frame = {}'.format(
            int(shape_G[0]), int(shape_G[1]), int(shape_G[2]), z_frame)

        # plot frame
        fig = plt.figure(figsize=(15, 15))
        plt.imshow(img, cmap='gray')
        plt.title(title)
        
        # create filename
        fname = plot_filebasename + '_z={}'.format(z_frame)

        # save image to file
        if img_format == ImgFrmt.SVG:
            fig.savefig(str(os.path.join(plot_path, fname) + '.svg'), format = 'svg',
                        dpi = 1200, bbox_inches = 'tight', pad_inches = 0)

        elif img_format == ImgFrmt.EPS:
            fig.savefig(str(os.path.join(plot_path, fname) + '_black.eps'), format = 'eps', dpi = 400,
                        bbox_inches = 'tight', pad_inches = 0)

        elif img_format == ImgFrmt.TIFF:
            png1 = BytesIO()
            fig.savefig(png1, format='png')
            png2 = Image.open(png1)
            png2.save((str(os.path.join(plot_path, fname) + '.tiff')))
            png1.close()

        plt.close(fig)

    return plot_path


def plot_histogram(x, xlabel = '', ylabel = '', bins = 100, _save = True, filepath = None,
                   xlabelfontsize = 20, ylabelfontsize = 20, xticksfontsize = 16, yticksfontsize = 16):
    
    ''' FUNCTION DESCRIPTION______________________________________________

    - plot data histogram

    - PARAMETERS
      x:      data to plot       (data type: numpy array)   
      xlabel: x axis label       (data type: string)
      ylabel: y axis label       (data type: string)
      bins:   number of bins     (data type: int)
      _save:  save flag          (data type: boolean)
      filepath: output path      (data type: string; example: '/a/b/filename.tiff' or '/a/b/filename')
      xlabelfontsize: font size  (data type: int)
      ylabelfontsize: font size  (data type: int)
      xticksfontsize: ticks size (data type: int)
      yticksfontsize: ticks size (data type: int)
    
    - return: filepath of the saved image (None otherwise) '''
    

    # plot histogram
    fig = plt.figure(tight_layout = True, figsize = (15, 15))
    plt.xlabel(xlabel, fontsize = xlabelfontsize)
    plt.ylabel(ylabel, fontsize = ylabelfontsize)
    plt.xticks(fontsize = xticksfontsize)
    plt.yticks(fontsize = yticksfontsize)
    plt.hist(x, bins = bins)

    # save histogram to file
    if _save:
        png1 = BytesIO()
        fig.savefig(png1, format = 'png')
        png2 = Image.open(png1)

        # add .tiff extension if missing
        if filepath.split('.')[-1] != '.tiff' or filepath.split('.')[-1] != '.tif':
            filepath = filepath + '.tiff'

        png2.save(filepath)
        png1.close()

        return filepath

    return None


def normalize(img, max_value=255.0, dtype=np.uint8):

    ''' FUNCTION DESCRIPTION______________________________________________

    - normalize image data

    - PARAMETERS
      img:       input image            (data type: dtype)
      max_value: max gray level value   (data type: double)
      dtype:     data type

    - return: normalized image

    '''

    max_v = img.max()
    min_v = img.min()

    if max_v != 0:

        if max_v != min_v:

            return (((img - min_v) / (max_v - min_v)) * max_value).astype(dtype)

        else:

            return ((img / max_v) * max_value).astype(dtype)

    else:

        return img.astype(dtype)


def print_info(X, text = '', _std = False, _return = False):

    ''' FUNCTION DESCRIPTION______________________________________________

    - print image info to screen

    - PARAMETERS
      X:        input image                     (data type: dtype)
      text:     text to be integrated           (data type: string)
      _std:     compute standard deviation flag (data type: boolean)
      _return:  output to list flag             (data type: boolean)
    
    '''


    if X is None:
        return None

    info = list()

    info.append(text)
    info.append(' * Image data type:    {}'.format(X.dtype))
    info.append(' * Image shape:        {}'.format(X.shape))
    info.append(' * Image max value:    {}'.format(X.max()))
    info.append(' * Image min value:    {}'.format(X.min()))
    info.append(' * Image mean value:   {}'.format(X.mean()))

    if _std:
        info.append(' * Image std value: {}'.format(X.std()))

    if _return:
        return info

    else:
        for l in info:
            print(l)
