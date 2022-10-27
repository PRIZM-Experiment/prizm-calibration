import sqlite3
import itertools
import collections
import os
import json
import pickle
import numpy as np
try:
    import scio.scio as scio
except:
    import scio
from metadatabase import __file__ as _path


# Global objects containing the directory addresses of: this module, the metadata, and PRIZM data.
_path = os.path.dirname(_path)
_directories = json.load(open(_path + '/settings.json', 'r'))

def source():
    """ Sources the data and metadata directories from the 'settings.json' file edited by the user. """

    # Loads the data and metadata directories from the 'settings.json' file.
    _directories = json.load(open(_path + '/settings.json', 'r'))

    return

def execute(query):
    """ Executes the input query against the metadatabase. """

    # Connects to the MDB and initializes a cursor.
    connection = sqlite3.connect(_directories['metadata'] + '/metadatabase.db')
    cursor = connection.cursor()

    # Queries the MDB.
    result_set = cursor.execute(query).fetchall()

    # Terminates the connection to the MDB.
    connection.commit()
    connection.close()

    return result_set

def count(categories=['Antenna', 'Switch', 'Temperature'], instruments=['100MHz', '70MHz'], channels=['EW', 'NS'], intervals=[(1524400000.0,1524500000.0),], quality=[1, 0, 'NULL'], integrity=[1, 0, 'NULL'], completeness=[1, 0, 'NULL']):
    """ Collects the classification, extension, alias, data type, and count of each file of the same type matching the input arguments. """

    # Initializes the SQLite query.
    query = ""

    # Builds the query from all possible combinations of the input arguments.
    for combination, (category, instrument, channel, (start, stop), quality, integrity, completeness) in enumerate(itertools.product(*[categories, instruments, channels, intervals, quality, integrity, completeness])):
        query = ("SELECT ArrayElements.element_name AS classification_name, "
                 "       CASE "
                 "            WHEN DataTypes.file_name LIKE '%.scio%' THEN 'scio' "
                 "            WHEN DataTypes.file_name LIKE '%.raw%' THEN 'raw' "
                 "       END "
                 "       AS file_extension, "
                 "       CASE "
                 "            WHEN DataTypes.file_alias LIKE 'time\_sys\_%' ESCAPE '\\' THEN ChannelOrientations.orientation_name "
                 "            WHEN DataTypes.file_alias LIKE 'pol%' THEN ChannelOrientations.orientation_name "
                 "            WHEN DataTypes.file_alias LIKE 'temp\_70[ABNE]%' ESCAPE '\\' THEN ChannelOrientations.orientation_name "
                 "            WHEN DataTypes.file_alias LIKE 'temp\_100[ABNE]%' ESCAPE '\\' THEN ChannelOrientations.orientation_name "
                 "            WHEN DataTypes.file_alias LIKE 'time\_%\_sys\_therms' ESCAPE '\\' THEN ChannelOrientations.orientation_name "
                 "            WHEN DataCategories.category_name = 'Switch' THEN DataCategories.category_name "
                 "            ELSE 'Housekeeping' "
                 "       END "
                 "       AS subclassification_name, "
                 "       CASE "
                 "            WHEN DataTypes.file_alias LIKE 'pol%' THEN 'pol' "
                 "            WHEN DataTypes.file_alias LIKE 'temp\_100[AB]%' ESCAPE '\\' THEN SUBSTR(DataTypes.file_alias, 1, 8) || SUBSTR(DataTypes.file_alias, 10, LENGTH(DataTypes.file_alias)) "
                 "            WHEN DataTypes.file_alias LIKE 'temp\_100[NE]%' ESCAPE '\\' THEN SUBSTR(DataTypes.file_alias, 1, 8) || SUBSTR(DataTypes.file_alias, 11, LENGTH(DataTypes.file_alias)) "
                 "            WHEN DataTypes.file_alias LIKE 'temp\_70[AB]%' ESCAPE '\\' THEN SUBSTR(DataTypes.file_alias, 1, 7) || SUBSTR(DataTypes.file_alias, 9, LENGTH(DataTypes.file_alias)) "
                 "            WHEN DataTypes.file_alias LIKE 'temp\_70[NE]%' ESCAPE '\\' THEN SUBSTR(DataTypes.file_alias, 1, 7) || SUBSTR(DataTypes.file_alias, 10, LENGTH(DataTypes.file_alias)) "
                 "            ELSE DataTypes.file_alias "
                 "       END "
                 "       AS file_alias, "
                 "       DataTypes.data_type AS data_type, "
                 "       DataDirectories.data_directory "
                 "FROM   DataDirectories "
                 "JOIN   DataCategories "
                 "ON     DataDirectories.data_category = DataCategories.data_category "
                f"AND    DataCategories.category_name = '{category}' "
                f"AND    DataDirectories.directory_completeness IS {completeness} "
                 "JOIN   DataFiles "
                 "ON     DataDirectories.data_directory = DataFiles.data_directory "
                f"AND    DataDirectories.time_start <= {stop} "
                f"AND    DataDirectories.time_stop >= {start} "
                f"AND    DataFiles.data_integrity IS {integrity} "
                f"AND    DataFiles.data_quality IS {quality} "
                 "JOIN   DataTypes "
                 "ON     DataFiles.data_file = DataTypes.data_file "
                 "JOIN   ChannelGroups "
                 "ON     DataFiles.channel_group = ChannelGroups.channel_group "
                 "JOIN   ChannelOrientations "
                 "ON     ChannelOrientations.channel_orientation = ChannelGroups.channel_orientation "
                f"AND    ChannelOrientations.orientation_name = '{channel}' "
                 "JOIN   ArrayElements "
                 "ON     ArrayElements.array_element = ChannelGroups.array_element "
                f"AND    ArrayElements.element_name = '{instrument}' ") + "UNION "*bool(combination) + query

    query = "SELECT classification_name, subclassification_name, file_extension, file_alias, data_type, COUNT(file_alias) FROM (" + query + ") GROUP BY classification_name, subclassification_name, file_alias"

    # Queries the MDB.
    result_set = execute(query)

    return result_set

def locate(categories=['Antenna', 'Switch', 'Temperature'], instruments=['100MHz', '70MHz'], channels=['EW', 'NS'], intervals=[(1524400000.0,1524500000.0),], quality=[1, 0, 'NULL'], integrity=[1, 0, 'NULL'], completeness=[1, 0, 'NULL']):
    """ Collects the classification, path, extension, alias, and data type of each file matching the input arguments. """

    # Initializes the SQLite query.
    query = ""

    # Builds the query from all possible combinations of the input arguments.
    for combination, (category, instrument, channel, (start, stop), quality, integrity, completeness) in enumerate(itertools.product(*[categories, instruments, channels, intervals, quality, integrity, completeness])):
        query = ("SELECT ArrayElements.element_name AS classification_name, "
                 "       DataDirectories.directory_address || '/' || DataTypes.file_name AS file_path, "
                 "       DataDirectories.time_start AS time_start, "
                 "       CASE "
                 "            WHEN DataTypes.file_name LIKE '%.scio%' THEN 'scio' "
                 "            WHEN DataTypes.file_name LIKE '%.raw%' THEN 'raw' "
                 "       END "
                 "       AS file_extension, "
                 "       CASE "
                 "            WHEN DataTypes.file_alias LIKE 'time\_sys\_%' ESCAPE '\\' THEN ChannelOrientations.orientation_name "
                 "            WHEN DataTypes.file_alias LIKE 'pol%' THEN ChannelOrientations.orientation_name "
                 "            WHEN DataTypes.file_alias LIKE 'temp\_70[ABNE]%' ESCAPE '\\' THEN ChannelOrientations.orientation_name "
                 "            WHEN DataTypes.file_alias LIKE 'temp\_100[ABNE]%' ESCAPE '\\' THEN ChannelOrientations.orientation_name "
                 "            WHEN DataTypes.file_alias LIKE 'time\_%\_sys\_therms' ESCAPE '\\' THEN ChannelOrientations.orientation_name "
                 "            WHEN DataCategories.category_name = 'Switch' THEN DataCategories.category_name "
                 "            ELSE 'Housekeeping' "
                 "       END "
                 "       AS subclassification_name, "
                 "       CASE "
                 "            WHEN DataTypes.file_alias LIKE 'pol%' THEN 'pol' "
                 "            WHEN DataTypes.file_alias LIKE 'temp\_100[AB]%' ESCAPE '\\' THEN SUBSTR(DataTypes.file_alias, 1, 8) || SUBSTR(DataTypes.file_alias, 10, LENGTH(DataTypes.file_alias)) "
                 "            WHEN DataTypes.file_alias LIKE 'temp\_100[NE]%' ESCAPE '\\' THEN SUBSTR(DataTypes.file_alias, 1, 8) || SUBSTR(DataTypes.file_alias, 11, LENGTH(DataTypes.file_alias)) "
                 "            WHEN DataTypes.file_alias LIKE 'temp\_70[AB]%' ESCAPE '\\' THEN SUBSTR(DataTypes.file_alias, 1, 7) || SUBSTR(DataTypes.file_alias, 9, LENGTH(DataTypes.file_alias)) "
                 "            WHEN DataTypes.file_alias LIKE 'temp\_70[NE]%' ESCAPE '\\' THEN SUBSTR(DataTypes.file_alias, 1, 7) || SUBSTR(DataTypes.file_alias, 10, LENGTH(DataTypes.file_alias)) "
                 "            ELSE DataTypes.file_alias "
                 "       END "
                 "       AS file_alias, "
                 "       DataTypes.data_type AS data_type "
                 "FROM   DataDirectories "
                 "JOIN   DataCategories "
                 "ON     DataDirectories.data_category = DataCategories.data_category "
                f"AND    DataCategories.category_name = '{category}' "
                f"AND    DataDirectories.directory_completeness IS {completeness} "
                 "JOIN   DataFiles "
                 "ON     DataDirectories.data_directory = DataFiles.data_directory "
                f"AND    DataDirectories.time_start <= {stop} "
                f"AND    DataDirectories.time_stop >= {start} "
                f"AND    DataFiles.data_integrity IS {integrity} "
                f"AND    DataFiles.data_quality IS {quality} "
                 "JOIN   DataTypes "
                 "ON     DataFiles.data_file = DataTypes.data_file "
                 "JOIN   ChannelGroups "
                 "ON     DataFiles.channel_group = ChannelGroups.channel_group "
                 "JOIN   ChannelOrientations "
                 "ON     ChannelOrientations.channel_orientation = ChannelGroups.channel_orientation "
                f"AND    ChannelOrientations.orientation_name = '{channel}' "
                 "JOIN   ArrayElements "
                 "ON     ArrayElements.array_element = ChannelGroups.array_element "
                f"AND    ArrayElements.element_name = '{instrument}' ") + "UNION "*bool(combination) + query

    query = "SELECT classification_name, subclassification_name, file_path, file_extension, file_alias, data_type FROM (" + query + ") ORDER BY classification_name, subclassification_name, time_start"

    # Queries the MDB.
    result_set = execute(query)

    return result_set

def _load(count_result_set, locate_result_set, parent_directory):
    """ Loads all files matching the result sets outputted by the `count` and `locate` functions. """

    # Initializes auxiliary data-loading dictionaries.
    data = collections.defaultdict(lambda: collections.defaultdict(dict))
    rows = collections.defaultdict(lambda: collections.defaultdict(dict))
    counter = collections.defaultdict(lambda: collections.defaultdict(dict))

    # Allocates the auxiliary data-loading dictionaries.
    for (classification_name, subclassification_name, _, file_alias, _, file_count) in count_result_set:
        data[classification_name][subclassification_name][file_alias] = [None]*file_count
        rows[classification_name][subclassification_name][file_alias] = [0] + [None]*file_count
        counter[classification_name][subclassification_name][file_alias] = 0

    # Loads each file matching the input arguments.
    for (classification_name, subclassification_name, file_path, file_extension, file_alias, data_type) in locate_result_set:
        index = counter[classification_name][subclassification_name][file_alias]

        if file_extension == 'scio':
            data[classification_name][subclassification_name][file_alias][index] = scio.read(parent_directory + file_path)
        if file_extension == 'raw':
            data[classification_name][subclassification_name][file_alias][index] = np.fromfile(parent_directory + file_path, data_type)

        rows[classification_name][subclassification_name][file_alias][index+1] = rows[classification_name][subclassification_name][file_alias][index] + len(data[classification_name][subclassification_name][file_alias][index])
        counter[classification_name][subclassification_name][file_alias] += 1

    # Initializes the output dictionary.
    output = collections.defaultdict(lambda: collections.defaultdict(dict))

    # Allocates and populates the output dictionary.
    for (classification_name, subclassification_name, file_extension, file_alias, data_type, _) in count_result_set:
        if subclassification_name == '':
            if file_extension == 'scio':
                output[classification_name][file_alias] = np.empty((rows[classification_name][subclassification_name][file_alias][-1], data[classification_name][subclassification_name][file_alias][-1].shape[1]), data_type)
            if file_extension == 'raw':
                output[classification_name][file_alias] = np.empty((rows[classification_name][subclassification_name][file_alias][-1],), data_type)

            for file_data, file_rows in zip(data[classification_name][subclassification_name][file_alias], rows[classification_name][subclassification_name][file_alias]):
                output[classification_name][file_alias][file_rows:file_rows+len(file_data)] = file_data

        else: 
            if file_extension == 'scio':
                output[classification_name][subclassification_name][file_alias] = np.empty((rows[classification_name][subclassification_name][file_alias][-1], data[classification_name][subclassification_name][file_alias][-1].shape[1]), data_type)
            if file_extension == 'raw':
                output[classification_name][subclassification_name][file_alias] = np.empty((rows[classification_name][subclassification_name][file_alias][-1],), data_type)

            for file_data, file_rows in zip(data[classification_name][subclassification_name][file_alias], rows[classification_name][subclassification_name][file_alias]):
                output[classification_name][subclassification_name][file_alias][file_rows:file_rows+len(file_data)] = file_data

    return output

def load(categories=['Antenna', 'Switch', 'Temperature'], instruments=['100MHz', '70MHz'], channels=['EW', 'NS'], intervals=[(1524400000.0,1524500000.0),], quality=[1, 0, 'NULL'], integrity=[1, 0, 'NULL'], completeness=[1, 0, 'NULL'], selection=None):
    """ Loads all data files matching the input arguments. """

    if selection == None:
        # Generates the needed result sets from the input arguments.
        count_result_set = count(categories, instruments, channels, intervals, quality, integrity, completeness)
        locate_result_set = locate(categories, instruments, channels, intervals, quality, integrity, completeness)
    else:
        # Loads the needed result sets from the input pickle file.
        count_result_set, locate_result_set = pickle.load(open(selection, 'rb'))

    return _load(count_result_set, locate_result_set, _directories['data'])

# ##################### OLD ##########################
#
# def read_scio_file(dirs, file_name, verbose):
#     """ Reads '.scio' files located in a given list of directories.
#
#     Looks for files with the given `file_name` in the input list of directories
#     `dirs`. If the file has been located in the provided directory, the function
#     attempts to read it. In case the file cannot be found and/or read, an error
#     message is printed. All files which have been successfully located and read
#     are stacked and returned as a single NumPy array.
#     (This function is largely equivalent to `prizmtools.read_pol_fast`).
#
#     Args:
#         dirs: a list of strings specifying the directories where the '.scio'
#             files of interest are stored.
#         file_name: a string in the format '*.scio' specifying the name of the
#             file of interest.
#         verbose: a boolean parameter which instructs the function to output
#             messages as the data is read when `True`, or to output no messages
#             when `False`.
#
#     Returns:
#         A NumPy array containing the information encapsulated in all files named
#         `file_name` stored in the directories `dirs`. If no files with the input
#         `file_name` can be found and/or read, an empty NumPy array is returned.
#     """
#
#     # Checks whether `dirs` is a single string. If that is the case, it is
#     # converted into a list with the provided string as its single entry. This
#     # guarantees compability with the rest of code that follows.
#     if isinstance(dirs, str):
#         dirs = [dirs]
#
#     # Generates a list which appropriately concatenates the string `file_name`
#     # to all string entries in the input list of directories `dirs`.
#     file_list = [d + '/' + f for d, f in zip(dirs, [file_name]*len(dirs))]
#
#     # Reads the '.scio' file into `scio_data_list`, which is a list of NumPy
#     # arrays with each array corresponding to a different entry in `dirs`. This
#     # operation is timed in case `verbose = True`.
#     read_start = time.time()
#     scio_data_list = scio.read_files(file_list)
#     read_end = time.time()
#
#     # Verbose message.
#     if verbose:
#         print(
#               '`read_scio_file`: operation `scio.read_files` lasted ',
#               read_end - read_start,
#               's.',
#               )
#
#     # Checks whether any file has not been found and/or read, making it feature
#     # in `scio_data_list` as a `None` entry. The indices associated with such
#     # files are stored in the list `indices`.
#     indices = []
#     for index, entry in enumerate(scio_data_list):
#         # The entries in `scio_data_list` can be either NumPy arrays or `None`.
#         # Since these objects are very different from each other, it is hard to
#         # make simple comparisons between them without running into errors.
#         # Thus, we must use 'try/except' to prevent the function from hiccuping
#         # at this step.
#         try:
#             # If files could not be found and/or read, the index associated with
#             # such files are stored in `indices`, and the path to that files is
#             # printed along with a warning message.
#             if entry is None:
#                 indices.append(index)
#                 # Verbose message.
#                 if verbose:
#                     print(
#                           'Could not find and/or read file: '
#                           + dirs[index] + '/' + file_name
#                           )
#         except ValueError:
#             continue
#
#     # Keeps only those entries of `scio_data_list` which are not `None`, i.e.,
#     # keeps all elements which do not feature in the list of `indices` generated
#     # above.
#     scio_data_list = [
#                       entry
#                       for index, entry in enumerate(scio_data_list)
#                       if index not in indices
#                       ]
#
#     # Attempts to stack the entries of `scio_data_list` into a single NumPy
#     # array `scio_data` using `numpy.vstack`. If this operation fails, it means
#     # `scio_data_list` is empty, in which case a warning message is printed and
#     # `scio_data` is assigned an empty NumPy array.
#     try:
#         scio_data = np.vstack(scio_data_list)
#     except ValueError:
#         scio_data = np.array([])
#         print('No files named `' + file_name + '` could be found and/or read.')
#
#     # Returns the `scio_data`.
#     return scio_data
#
#
# def read_raw_file(dirs, file_name, dtype, verbose):
#     """ Reads '.raw' files located in a given list of directories.
#
#     Looks for files with the given `file_name` in the input list of directories
#     `dirs`. If the file has been located in the provided directory, the function
#     attempts to read it. In case the file cannot be found and/or read, an error
#     message is printed. All files which have been successfully located and read
#     are stacked and returned a single NumPy array.
#     (This function is largely equivalent to `prizmtools.read_field_many_fast`).
#
#     Args:
#         dirs: a list of strings specifying the directories where the '.raw' files
#             of interest are stored.
#         file_name: a string in the format '*.raw' specifying the name of the
#             file of interest.
#         verbose: a boolean parameter which instructs the function to output
#             messages as the data is read when `True`, or to output no messages
#             when `False`.
#         dtype: the desired data type to be returned, defaulted to be 'float64'.
#
#     Returns:
#         A NumPy array containing the information encapsulated in all files named
#         `file_name` stored in the directories `dirs`. If no files with the input
#         `file_name` can be found and/or read, an empty NumPy array is returned.
#     """
#
#     # Checks whether `dirs` is a single string. If that is the case, it is
#     # converted into a list with the provided string as its single entry. This
#     # guarantees compability with the rest of code that follows.
#     if isinstance(dirs, str):
#         dirs = [dirs]
#
#     # Reads the '.raw' file into `raw_data_list`, which is a list of NumPy
#     # arrays with each array corresponding to a different entry in `dirs`. This
#     # operation is timed in case `verbose = True`. In case a file has not been
#     # found and/or read, its corresponding index (i.e., its position in the
#     # directory list) is stored in the list `indices`.
#     indices = []
#     raw_data_list = []
#     read_start = time.time()
#     for index, dir in enumerate(dirs):
#         try:
#             raw_data_list.append(
#                                  np.fromfile(dir + '/' + file_name, dtype=dtype)
#                                  )
#         except:
#             indices.append(index)
#     read_end = time.time()
#
#     # Verbose message.
#     if verbose:
#         print(
#               '`read_raw_file`: operation `numpy.fromfile` lasted ',
#               read_end - read_start,
#               's.',
#               )
#
#         # If files could not be found and/or read, the index associated with such
#         # file, which is stored in `indices`, is used to print their path along with
#         # a warning message.
#         for index in indices:
#             print(
#                   'Could not find and/or read file: '
#                   + dirs[index] + '/' + file_name
#                   )
#
#     # Attempts to stack the entries of `raw_data_list` into a single NumPy array
#     # `raw_data` using `numpy.hstack`. If this operation fails, it means
#     # `raw_data_list` is empty, in which case a warning message is printed and
#     # `raw_data` is assigned an empty NumPy array.
#     try:
#         raw_data = np.hstack(raw_data_list)
#     except ValueError:
#         raw_data = np.array([])
#         print('No files named `' + file_name + '` could be found and/or read.')
#
#     # Returns the `raw_data`.
#     return raw_data
