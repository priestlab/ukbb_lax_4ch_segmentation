from os.path import isdir
from os.path import dirname
from os import makedirs

def secure_folder(folder):
    if not isdir(folder):
        makedirs(folder)


def secure_filepath(filepath):
    """ Make sure the folder of filepath is secured """
    dirpath = dirname(filepath)

    if not isdir(dirpath):
        if dirpath is not '':
            makedirs(dirpath)

