import pydicom
import numpy as np
from zipfile import ZipFile
from pydicom.filebase import DicomBytesIO

from .bsa import haycock_BSA



def load_zip_data(zip_file, series_descriptions=None, return_array=False):
    """
    Load DCM files or NUMPY arrays from zip_file

    Params
    ------
    zip_file            : the file path to the zip file
                        - e.g. /directory/xxxxxxx_20208_2_0.zip
    series_descriptions : the series to extract from the zip file
                        - str or list of strs, e.g. "CINE_segmented_LAX_4Ch"
    return_array        : bool, whether to return arrays or dcms

    Return
    ------
    data                : arrays or dcms in the format of Dict
    """

    zf = ZipFile(zip_file, "r")
    content = zf.namelist()
    dcms = [pydicom.dcmread(DicomBytesIO(zf.read(x)))
            for x in content if x.endswith('.dcm')]

    SAX = False
    if series_descriptions == None:
        series_descriptions = list(set([x.SeriesDescription for x in dcms]))
    elif series_descriptions == "CINE_segmented_SAX":
        series_descriptions = list(set([x.SeriesDescription for x in dcms]))
        series_descriptions = sorted([x for x in series_descriptions if 'CINE_segmented_SAX_b' in x], 
                                     key=lambda x: int(x.split("_")[-1].replace("b","")))
        SAX = True
    elif isinstance(series_descriptions, str):
        series_descriptions = [series_descriptions]

    data = {}
    for series_description in series_descriptions:
        series_dcms = [x for x in dcms if
                       x.SeriesDescription == series_description]
        if len(set([x.SeriesInstanceUID for x in series_dcms])) > 1:
            series_dcms = [x for x in series_dcms if
                           x.SeriesInstanceUID == series_dcms[0].SeriesInstanceUID]

        series_dcms = sorted(series_dcms, key=lambda x: int(x.InstanceNumber))
        if return_array:
            data.update({series_description: np.array([x.pixel_array for x in series_dcms])})
        else:
            data.update({series_description: series_dcms})

    if SAX:
        if return_array:
            data = {"CINE_segmented_SAX": np.array([data[x] for x in series_descriptions])}
        else:
            data = {"CINE_segmented_SAX": [data[x] for x in series_descriptions]}
        series_descriptions = ["CINE_segmented_SAX"]


    if len(series_descriptions) == 1:
        data = data[series_descriptions[0]]

    return data



def load_zip_meta(zip_file, series_description, meta, by_frame=False):
    """ 
    Load DCM Meta from zip_file 

    PatientSize/PatientSex/PatientWeight/PatientAge/PixelSpacing
    SeriesDescription/SeriesInstanceUID/SeriesNumber/InstanceNumber
    """
    dcms = load_zip_data(zip_file, series_description, return_array=False)

    if by_frame:
        dcm_meta = np.array([getattr(dcm, meta) for dcm in dcms])
    else:
        dcm_meta = getattr(dcms[0], meta)

    return dcm_meta


def load_zip_pixelspacing(zip_file, series_description, by_frame=False):
    """ 
    Load DCM pixel_spacing from zip_file 
    """

    pixelspacing = load_zip_meta(zip_file, series_description, 'PixelSpacing', by_frame=by_frame)
    pixelspacing = np.array(pixelspacing).astype(np.float)

    return pixelspacing
    

def load_zip_BSA(zip_file, series_description):
    """ 
    Load DCM BSA from zip_file 
    """

    height = float(load_zip_meta(zip_file, series_description, 'PatientSize'))
    weight = float(load_zip_meta(zip_file, series_description, 'PatientWeight'))
    BSA = haycock_BSA(height*100.0, weight)

    return BSA


