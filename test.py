from pathlib import Path
import unittest
from copy import deepcopy

import cv2
import numpy as np

from features import luminance_feature, construct_feature_vector, dimension_flat_for_ann
from main import discover_image_dirs,  get_images_in_dir, create_img_obj
from pyramid import create_arbitrary_pyramid, construct_pyramid, reverse_pyramid


images_path = Path('images') / 'input'


class TestDirectoryScope(unittest.TestCase):

    def setUp(self):
        self.directories = None
        self.image_dirs = None
        self.image_list = []
        pass

    def tearDown(self):
        pass

    def test_get_directory(self):
        self.directories = discover_image_dirs()
        assert type(self.directories) == list
        assert len(self.directories) >= 1

    def test_get_images_in_dir(self):
        self.directories = [Path('images') / 'input' / 'test']
        self.image_dirs = get_images_in_dir(self.directories[0])
        assert type(self.image_dirs) == list
        assert len(self.image_dirs) == 3

    def test_create_img_obj(self):
        self.directories = [Path('images') / 'input' / 'test']
        self.image_dirs = get_images_in_dir(self.directories[0])
        for img_dir in self.image_dirs:
            self.image_list.append(create_img_obj(img_dir))
        assert len(self.image_list) == 3
        assert self.image_list[0].__name__ == 'Image'


class TestImageLoading(unittest.TestCase):

    def setUp(self):
        self.directory = discover_image_dirs()
        self.image_dirs = get_images_in_dir(self.directory[0])
        self.image_list = [create_img_obj(image_dir) for image_dir in self.image_dirs]

    def tearDown(self):
        pass

    def test_load_images(self):
        for image in self.image_list:
            image.load_image_values()
        assert all(map(lambda x: x is not None, self.image_list))

    def test_define_metadata(self):
        for image in self.image_list:
            image.define_metadata()
        assert all(map(lambda x: x.metadata in str(x.path), self.image_list))


class TestFeature(unittest.TestCase):

    def setUp(self):
        # self.curr_path =
        # self.A = Image('A.jpg')
        # self.A_p = Image('A_p.jpg')
        # self.B = Image('B.jpg')
        #
        # if any(map(lambda x: x is None,
        #            [self.A, self.A_p, self.B])):
        #     raise IOError("Error, samples image not found.")
        self.layers = 5
        self.image = cv2.imread('test.png')
        self.yiq = luminance_feature(self.image)
        pyramid = {'A': [self.yiq[:,:,0]], 'A_p': [self.yiq[:,:,0]], 'B': [self.yiq[:,:,0]]}
        self.pyramids = construct_pyramid(pyramid, self.layers-1)
        pass

    def tearDown(self):
        pass

    def test_luminancce_feature(self):
        yiq = luminance_feature(self.image)
        assert yiq[0, 0, 0] != self.image[0, 0, 0]
        assert yiq.shape == self.image.shape

    def test_normalize_img_to_float(self):
        pass

    def test_construct_feature_vector(self):
        feat_a = construct_feature_vector(self.pyramids['A'])
        assert type(feat_a) == list
        assert len(feat_a) == self.layers

    def test_dimension_flat_for_ann(self):
        feat_a = construct_feature_vector(self.pyramids['A'])
        flat_feat_a = dimension_flat_for_ann(feat_a[0])
        first_h, first_w = feat_a[0].shape[0:2]
        assert flat_feat_a.shape[0] == first_h*first_w


class TestPyramid(unittest.TestCase):

    def setUp(self):
        self.image = cv2.imread('test.png')
        self.pyramid = {'A': [self.image]}

    def tearDown(self):
        pass

    def test_construct_pyramid(self):
        resulting_pyr = construct_pyramid(self.pyramid, 4)
        assert len(resulting_pyr['A']) == 5

    def test_reverse_pyramid(self):
        pass


if __name__ == '__main__':
    unittest.main()
