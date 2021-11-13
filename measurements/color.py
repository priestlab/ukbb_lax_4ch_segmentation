import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

cust_colors = [(225, 0, 0),
               (0, 225, 0),
               (225, 10, 150),
               (0, 0, 225),
               (0, 100, 250),
               (250, 100, 0),
               (225, 0, 225),
               (100, 225, 0),
               (10, 225, 225)]

custom_cmap = ListedColormap(np.concatenate([np.array(cust_colors, dtype=np.float)/255.,
                                             np.ones((len(cust_colors),1))], axis=1))


class plt_color_wheel(object):
    def __init__(self, cmap=None, vmin=0.0, vmax=1.0):
        """
        Usage: cmap[None, 'Pastel1', ...]
        colorWheel = plt_color_wheel(cmap='Pastel1', vmin=0, vmax=1)
        colorWheel.get_RGB_wheel()
        >>>array([[251, 180, 174],
        >>>       [179, 205, 227],
        >>>       [204, 235, 197],
        >>>       [222, 203, 228],
        >>>       [255, 255, 204],
        >>>       [229, 216, 189],
        >>>       [253, 218, 236],
        >>>       [242, 242, 242]], dtype=uint8)
        
        colorWheel.get_RGB(0.2)
        >>>array([179, 205, 227], dtype=uint8)
        """
        self.cmap_name = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else custom_cmap
        self.norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
    
    def scale_uint8(self, colors):
        assert np.max(colors) <= 1 and np.min(colors) >= 0, \
        f"Color input should have value ranging between 0 and 1: [{colors.min()}: {colors.max()}]"
        return (np.array(colors) * 255.).astype(np.uint8)
    
    def get_RGB_wheel(self, n_colors=8):
        x = np.linspace(0.0, 1.0, n_colors)
        wheel = self.cmap(x)[:,:3]
        wheel = self.scale_uint8(wheel)
        return wheel

    def get_RGB(self, val):
        assert val <= self.vmax and val >= self.vmin, \
        f"The input value should be between vmin and vmax: [{self.vmin}, {self.vmax}]"
        rgb = self.scalarMap.to_rgba(val)[:3]
        rgb = self.scale_uint8(rgb)
        return rgb
