import numpy as np

def read_vna_data(filename, delimiter=",", encoding="ISO-8859-1"):
    """
    Reads VNA measurements for efficiency calculation.
    2017 - .csv with ; delimiters
    2018 - .csv with , delimiters
    2019 - .csv with , delimiters
    2020 - Non-existent
    2021 - .txt with \t delimtiers

    Data from 2018 are .csv with peculiarities handled by this file.
    Data from 2022 are .txt with formatting as expected

    Original function was written by Kelly A. Foran, adapted by Ronniy C. Joseph.

    Parameters
    ----------
    filename
    startkey
    delimiter
    encoding

    Returns
    -------

    Written by Kelly A. Foran
    Adapted by Ronniy C. Joseph
    """
    #Loop through file to figure out where the header starts, and how many data rows we're reading and create appropiate
    #arrays
    header_line, n_data_rows = find_start_and_end(filename)
    data = np.zeros((n_data_rows, 6))

    read_labels = False
    read_data = False
    data_counter = 0

    with open(filename, 'r', encoding=encoding , errors = 'replace') as datafile:
        for index, line in enumerate(datafile):

            #Read VNA Measurement after the Column Header was read, see below.
            if read_data:
                data[data_counter] = decode_line(line, delimiter=delimiter, n_columns = len(column_header))
                data_counter +=1

            #Record Header Labels after the newline was found, see below.
            if read_labels:
                column_header = line.strip().split(delimiter)
                read_labels = False
                read_data = True

            #Try and find the first newline if you haven't found it already
            if read_labels != True and read_data != True:
                tags = line.split(delimiter)
                try:
                    if tags[0] == '\n':
                        read_labels = True
                except IndexError:
                    pass
    return data


def find_start_and_end(filename, lookup="\n", encoding="ISO-8859-1"):
    """
    This function reads a .txt file and returns the line where it finds the look-up character, and computes how many
    lines to the end of file

    It was built to scan Nivek's S11 VNA measurement outputs and figure out where the data actually starts so you
    can ignore all the metadata

    parameters
    -----------
    filename: str
        path to text file

    lookup: str
        string that marks end of metadata

    encoding: str
        type of encoding (utf-8, etc.)

    """
    header_line = None
    with open(filename, encoding=encoding) as datafile:
        line_counter = 0
        for num, line in enumerate(datafile):
            if lookup == line:
                header_line = num + 1
            line_counter += 1
    try:
        n_data_rows = line_counter - header_line - 1
    except TypeError:
        print(f"Couldn't find look-up character {lookup}. Are you sure the file actually contains this? ")
        raise
    return header_line, n_data_rows


def decode_line(line, delimiter, n_columns):
    #Deals with formatting challenges across the various data sets
    #The challenge is that frequency formatting changes based on year an delimiter, pending on channel spacing
    #There are two cases
    #All columns have two entries (pre- and post-comma)
    #Or the Frequency channel only has 1 entry, because there is no decimal comma

    tags = line.strip().split(delimiter)
    empty_column = int((len(tags) - 1) / 2)
    data = np.zeros(n_columns - 1)
    tags = tags[:empty_column] + tags[(empty_column + 1):]
    counter = 0

    #print(tags)
    #2021 VNA data, tab delimited data lines up with number of column labels
    if len(tags) == 6:
        for s, string in enumerate(tags):
            data[s] = string.replace(",", ".")
        data = data.astype(np.float)

    # <2021 data where every column has been split into before and after decimal points
    elif len(tags) == 12:
        indices = np.array([0,2,4,6,8,10])
        for s in indices:
            data[counter] = tags[s] + "." + tags[s+1]
            counter += 1
        data = data.astype(np.float)

    #<2021 where frequencies are integers, but other columns have been split into two
    #TODO build a robuster check there might be the unlikely odd case the Magnitude/Phase columns are ints
    elif len(tags) == 10:
        indices = np.array([0,1,3,5,6,8])
        for s in indices:
            if s == 0 or s == empty_column :
                data[counter] = tags[s]
            else:
                data[counter] = tags[s] + "." + tags[s+1]
            counter += 1
        data = data.astype(np.float)
    else:
        raise Exception("A new case of VNA data formatting!")
    return data

