import cv2

from features import luminance_feature


class Image:
    """
    Image object class.

    :params relative_path: Posix path object indicating where the image is.
    """
    def __init__(self, relative_path, **kwargs):
        self.__name__ = 'Image'
        self.__dict__.update(kwargs)
        self.path = relative_path
        self.pixels = None
        self.layers = []
        self.metadata = None
        self.yiq = None

    def load_image_values(self):
        # turns out cv2 doesnt take posix path objects.. haha
        full_path = self.path.resolve()
        self.pixels = cv2.imread(str(full_path))

    def define_metadata(self):
        split_path = str(self.path).split('/')
        self.metadata = split_path[-1].split('.')[0]

    def convert_rgb_to_yiq(self):
        self.yiq = luminance_feature(self.pixels)

    def normalize_img_to_float(self):
        self.pixels = self.pixels / 255.

    def convert_bgr_to_rgb(self):
        self.pixels = cv2.cvtColor(self.pixels, cv2.COLOR_BGR2RGB)

