import cv2
import pydicom
import numpy as np
import pandas as pd

from skimage.exposure import rescale_intensity
from scipy.ndimage.measurements import center_of_mass
from torch.utils.data import Dataset
from zipfile import ZipFile
from pydicom.filebase import DicomBytesIO


################################################################################################
# Function
# loading zip data
################################################################################################

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


################################################################################################
# Dataset Classes
# UKBB_MRI_Base
# UKBB_MRI_LAX_4Ch
################################################################################################

class UKBB_MRI_Base(Dataset):
    def resize(self, images, masks):
        resized_images = self.pad_data(images, (256, 256))
        resized_masks = self.pad_data(masks, (256, 256))

        ccom = center_of_mass(resized_images)[-2:]
        ccom = [int(x) for x in ccom]

        if min(self.image_size) <= 192:
            resized_images = self.crop_data(resized_images, ccom, self.image_size)
            resized_masks = self.crop_data(resized_masks, ccom, self.image_size)

        return resized_images, resized_masks

    @staticmethod
    def pad_data(data, shape=(224,224)):
        if data is None:
            return data

        assert data.ndim >= 2, \
        f"Data need to have at least 2 dimensions, data.shape: {data.shape}"

        padding = [(0,0) for i in range(data.ndim)]

        width, height = data.shape[-2:]
        w_pad = max(shape[0] - width, 0) // 2
        w_pad = (w_pad, max(shape[0] - width - w_pad, 0))
        h_pad = max(shape[1] - height, 0) // 2
        h_pad = (h_pad, max(shape[1] - height - h_pad, 0))

        padding[-1] = h_pad
        padding[-2] = w_pad

        padded_data = np.pad(data, padding, 'minimum')

        return padded_data

    @staticmethod
    def crop_data(data, ccom, shape):
        assert data.ndim >= 2, \
        f"Data need to have at least 2 dimensions, data.shape: {data.shape}"

        w0 = int(ccom[0] - shape[0]//2)
        h0 = int(ccom[1] - shape[1]//2)
        w1, h1 = shape

        return data[..., w0:w0+w1, h0:h0+h1]

    @staticmethod
    def rescale_intensity(data):
        if data.ndim > 2:
            data = np.array([UKBB_MRI_Base.rescale_intensity(frame) for frame in data])
        else:
            data = rescale_intensity(data, out_range=np.uint8).astype(np.uint8)

        return data


    @staticmethod
    def normalize(data):
        data = data.astype(np.float64)
        if data.ndim > 2:
            data = np.array([UKBB_MRI_Base.normalize(frame) for frame in data])
        else:
            #data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
            data = (data - data.mean()) / data.std()
        return data

    @staticmethod
    def hist_equalize(data):
        # Histogram Equalization
        if data.ndim > 2:
            data = np.array([UKBB_MRI_Base.hist_equalize(frame) for frame in data])
        else:
            clahe = cv2.createCLAHE(clipLimit=0.02)
            data = clahe.apply(data.astype(np.uint8))
        return data

    def preprocessing(self, image):
        image = self.rescale_intensity(image)
        image = self.hist_equalize(image)
        image = image.astype(np.float64)/255.
        return image



class UKBB_MRI_LAX_4Ch(UKBB_MRI_Base):
    def __init__(self, root_dir, csv_data="labels.csv",
                 image_dir="images", mask_dir="masks",
                 image_size=(256,256), num_classes=5,
                 image_type="RGB", image_format="zip",
                 image_preprocess=True, seed=1234, 
                 debug=False):

        self.root_dir           = root_dir
        self.image_dir          = image_dir or f"{root_dir}/images"
        self.mask_dir           = mask_dir or f"{root_dir}/masks"
        self.image_preprocess   = image_preprocess
        self.image_size         = image_size
        self.labels             = pd.read_csv(csv_data) if isinstance(csv_data, str) else csv_data
        self.image_type         = image_type.lower()
        self.image_format       = image_format
        self.num_classes        = num_classes
        np.random.seed(seed)

        if debug:
            np.random.shuffle(self.labels.values)
            self.labels = self.labels.iloc[0:debug]
        
    def __len__(self):
        return len(self.labels)

    def load_image(self, pid):
        """
        Loading in MRI images
        """
        if self.image_format == 'zip':
            image = load_zip_data(f"{self.image_dir}/{pid}.zip", 
                                  series_descriptions="CINE_segmented_LAX_4Ch",
                                  return_array=True)
            image = np.transpose(image, (0,2,1))
        else:
            image = np.load(f"{self.image_dir}/{pid}.npy")

        if self.image_preprocess:
            image = self.preprocessing(image)
        return image.astype(np.float64)
        
    def load_mask(self, pid):
        try:
            mask_path = f"{self.mask_dir}/{pid}.npy"
            mask = np.load(mask_path).astype(np.float64)
            return mask
        except:
            return None


    def __getitem__(self, idx):
        # Load in image and masks
        pid = self.labels.iloc[idx,0]
        image = self.load_image(pid)
        mask = self.load_mask(pid)
        
        # Resize image and mask into the desired image_size
        image, mask = self.resize(image, mask)
        
        # Normalizing the image
        image = self.normalize(image)

        # Convert image to RGB if needed
        if self.image_type == "rgb":
            if image.ndim == 3:
                image = np.repeat(np.expand_dims(image, 1), 3, 1)
            else:
                image = np.repeat(np.expand_dims(image, 0), 3, 0)

        
        return (image, mask, pid)


