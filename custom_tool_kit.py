import numpy as np
import math


# =================================== CLASSES ===================================

class Bcolors:
    VERB        = '\033[95m'
    OKBLUE      = '\033[94m'
    OKGREEN     = '\033[92m'
    WARNING     = '\033[93m'
    FAIL        = '\033[91m'
    ENDC        = '\033[0m'
    BOLD        = '\033[1m'
    UNDERLINE   = '\033[4m'


# =================================== METHODS ===================================

def extract_parameters(filename, param_names, _verb=False):
    
    ''' FUNCTION DESCRIPTION______________________________________________
    
    - read parameter values in filename.txt and export them in a dictionary
    
    - PARAMETERS
      filename:     text file name  (data type: string)
      param_names:  parameter names (data type: string list)
      _verb:        verbose flag    (data type: boolean)

    - RETURN: dictionary of parameters adopted for Structure Tensor Analysis
       
    '''

    # read parameter values from .txt file
    param_values = search_value_in_txt(filename, param_names)
    print('\n ***  Parameters : \n')

    # create dictionary of parameters
    parameters = {}
    for i, p_name in enumerate(param_names):

        parameters[p_name] = float(param_values[i])

        if _verb:
            print(' - {} : {}'.format(p_name, param_values[i]))

    if _verb: print('\n \n')

    return parameters


def write_on_txt(strings, txt_path, _print = False, mode = 'a'):

    ''' FUNCTION DESCRIPTION______________________________________________
    
    - write each element of the input string list into separate lines
      of the .txt file at txt_path 
      (if _print is True, strings are also printed to screen)

    - PARAMETERS
      strings:  list of strings                     (data type: list)
      txt_path: path of .txt file to be compiled    (data type: string)
      _print:   print flag                          (data type: boolean)
      mode:     file opening mode                   (data type: char)
          
    '''

    with open(txt_path, mode = mode) as txt:

        for s in strings:
            txt.write(s + '\n')

            if _print:
                print(s)


def all_words_in_txt(filepath):

    ''' FUNCTION DESCRIPTION______________________________________________
    
    - extract all words in the txt file and put them into a list
    
    '''     

    words = list()

    with open(filepath, 'r') as f:
        data = f.readlines()

        for line in data:

            for word in line.split():
                words.append(word)

    return words


def search_value_in_txt(filepath, strings_to_search):

    ''' FUNCTION DESCRIPTION______________________________________________

    - read the parameters names in the strings_to_search list, and extract the  
      corrisponednt parameters value from the .txt file at filepath

    - PARAMETERS
      filepath:             path of the .txt file   (data type: string)
      strings_to_search:    string or list of strings (parameters names) to be searched   
           
    '''

    # convert to list
    if type(strings_to_search) is not list:
        strings_to_search = [strings_to_search]

    # read all words in .txt file at filepath
    words = all_words_in_txt(filepath)

    # search strings
    values = [words[words.index(s) + 2] for s in strings_to_search if s in words]

    return values


def manage_path_argument(source_path):
   
    ''' FUNCTION DESCRIPTION______________________________________________

    - manage input parameters of Structure Tensor Analysis

      if called by terminal, source_path is a list including one string only (i.e. the correct source path)

    - PARAMETERS
      source_path: list including the paths of the image files to be processed (data type: string or list of strings)
      
    - WARNING:     do not include ' ' whitespaces!!!
      
    '''                       

    try:
        if source_path is not None:

            if type(source_path) is list:

                if len(source_path) > 1:                    
                    given_path = ' '.join(source_path)

                else:
                    given_path = source_path[0]

            else:
                given_path = source_path

            if given_path.endswith('/'):
                given_path = given_path[0:-1]

            return given_path

        else:
            return None

    except:
        print(Bcolors.FAIL + '[manage_path_argument] -> source_path empty?' + Bcolors.ENDC)

        return None


def pad_dimension(matrix, shape):

    ''' FUNCTION DESCRIPTION______________________________________________
    
    - check whether the input matrix shape is equal to 'shape':
      if not, pad every axis with zeroes
    
    - PARAMETERS
      matrix: input data        (data type: numpy array)
      shape:  numpy array shape (data type: numpy array)

    - RETURNS: zero-padded numpy array
      
      '''
      
    if matrix is not None and shape is not None:

        # check 0th axis
        if matrix.shape[0] < shape[0]:
            zero_pad = np.zeros((int(shape[0] - matrix.shape[0]), matrix.shape[1], matrix.shape[2]))
            matrix = np.concatenate((matrix, zero_pad), 0)

        # check 1st axis
        if matrix.shape[1] < shape[1]:
            zero_pad = np.zeros((matrix.shape[0], int(shape[1] - matrix.shape[1]), matrix.shape[2]))
            matrix = np.concatenate((matrix, zero_pad), 1)

        # check 2nd axis
        if matrix.shape[2] < shape[2]:
            zero_pad = np.zeros((matrix.shape[0], matrix.shape[1], int(shape[2] - matrix.shape[2])))
            matrix = np.concatenate((matrix, zero_pad), 2)

        return matrix

    else:
        raise ValueError('Data block or shape is None')


def create_coord_by_iter(r, c, z, shape_P, _z_forced = False):
    
    ''' FUNCTION DESCRIPTION______________________________________________

    - define initial coordinate of (data) parallelepiped inside the whole volume

    - PARAMETERS
      r: row index                                                    (data type: int >= 0)
      c: column index                                                 (data type: int >= 0)
      z: zeta index                                                   (data type: int >= 0)
      shape_P:   spatial resolution of orientatin analysis (pixel)    (data type: numpy array)
      _z_forced: flag to select manually z index                      (data type: boolean)

    '''
    
    row = r * shape_P[0]
    col = c * shape_P[1]

    if _z_forced:
        zeta = z
    else:
        zeta = z * shape_P[2]

    return (row, col, zeta)


def nextpow2(n):

    ''' FUNCTION DESCRIPTION______________________________________________

    - return the smallest power of 2 higher than n

    - PARAMETERS
      n: input number   (data type: int or float)

    '''

    return int(np.power(2, np.ceil(np.log2(n))))


def magnitude(x):

    ''' FUNCTION DESCRIPTION______________________________________________

    - return the magnitude of the input number

    - PARAMETERS
      x: input number   (data type: int or float)

    '''

    if x is None:
        return None

    if x < 0:
        return magnitude(abs(x))

    elif x == 0:
        return -1

    else:
        return int(math.floor(math.log10(x)))


def create_slice_coordinate(start_coord, shape_of_subblock):

    ''' FUNCTION DESCRIPTION______________________________________________

    - define slice coordinates to identify an array having shape = shape_of_subblock
      starting at start_coord

    - PARAMETERS
      start_coord:          slice initial coordinate (data type: tuple o array)
      shape_of_subblock:    sub-array shape          (data type: tuple o array)

    - RETURN: coordinates of selected sub-array

    '''
    
    selected_slice_coord = []
    for (start, s) in zip(start_coord, shape_of_subblock):
        selected_slice_coord.append(slice(start, start + s, 1))

    return selected_slice_coord


def seconds_to_min_sec(sec):
    
    ''' FUNCTION DESCRIPTION______________________________________________

    - convert seconds to different time unit if > 60 (e.g. hours, minutes)

    - PARAMETERS
      sec: input time in seconds    (data type: int)

    '''

    if sec < 60:
        return int(0), int(0), int(sec)

    elif sec < 3600:
        return int(0), int(sec // 60), int(sec % 60)

    else:
        h = int(sec // 3600)

        return h, int(sec // 60) - h * 60, int(sec % 60)

