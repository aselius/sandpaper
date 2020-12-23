import cv2
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
import pyflann

from config import Config

kernel = 1/273 * np.array([1, 4, 7, 4, 1,
                           4, 16, 26, 16, 4,
                           7, 26, 41, 26, 7,
                           4, 16, 26, 16, 4,
                           1, 4, 7, 4, 1])

kernel_lg = np.array([[0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902],
                      [0.01330621, 0.0596343 , 0.09832033, 0.0596343 , 0.01330621],
                      [0.02193823, 0.09832033, 0.16210282, 0.09832033, 0.02193823],
                      [0.01330621, 0.0596343 , 0.09832033, 0.0596343 , 0.01330621],
                      [0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902]])

kernel_sm = np.array([[0.01134374, 0.08381951, 0.01134374],
                      [0.08381951, 0.61934703, 0.08381951],
                      [0.01134374, 0.08381951, 0.01134374]])


def luminance_feature(img):
    """
    Converts BGR image data to YIQ image data

    :param img: BGR colorspace pixel data
    :return yiq: YIQ colorspace pixel data
    """
    # YIQ color space
    # [ Y ]     [ 0.299   0.587   0.114 ] [ R ]
    # [ I ]  =  [ 0.596  -0.275  -0.321 ] [ G ]
    # [ Q ]     [ 0.212  -0.523   0.311 ] [ B ]
    # https://www.cs.rit.edu/~ncs/color/

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = np.array([
        [0.299, 0.587, 0.114],
        [0.596, -0.275, -0.321],
        [0.212, -0.523, 0.311]
    ])
    return np.dot(rgb, transform)


def yiq_to_bgr(img):
    """
    Converts YIQ to BGR data

    :param img:
    :return:
    """
    # [ R ]     [ 1   0.956   0.621 ] [ Y ]
    # [ G ]  =  [ 1  -0.272  -0.647 ] [ I ]
    # [ B ]     [ 1  -1.105   1.702 ] [ Q ]
    # https://www.cs.rit.edu/~ncs/color/

    transform = np.array([
        [1., 0.956, 0.621],
        [1., -0.272, -0.647],
        [1., -1.105, 1.702]
    ])
    rgb = np.dot(img, transform)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def luminance_mapping():
    raise NotImplementedError


def pad_img(img, padding_option):
    return np.pad(img, padding_option, mode='symmetric')


def feature_vector(pyramid, drop=False):
    """

    :param pyramid:
    :return feature_vec:
    """
    feature_vec = [[]]
    conf = Config()
    for layer in range(1, len(pyramid)):
        l1 = pad_img(pyramid[layer - 1], conf.padding_l1)
        l2 = pad_img(pyramid[layer], conf.padding_l2)

        patches_l1 = extract_patches_2d(l1, (conf.nl1, conf.nl1))
        patches_l2 = extract_patches_2d(l2, (conf.nl2, conf.nl2))

        if drop:
            patches_l2 = patches_l2.reshape(patches_l2.shape[0], -1)[:, :int(3 * conf.nfine)]

        level_features = []
        h, w = pyramid[layer].shape[0], pyramid[layer].shape[1]
        for row in range(h):
            for col in range(w):
                level_features.append( np.hstack([
                    patches_l1[int(np.floor(row / 2.) * np.ceil(w / 2.) + np.floor(col / 2.))].flatten(),
                    patches_l2[row * w + col].flatten()
                ]))

        feature_vec.append(np.vstack(level_features))
    return feature_vec


def ann_index(pyramid_a, pyramid_ap):
    feat_a = feature_vector(pyramid_a)
    feat_ap = feature_vector(pyramid_ap, drop=True)

    layers = len(pyramid_a)

    flann = [pyflann.FLANN() for x in range(layers)]
    index_params = [[] for x in range(layers)]

    a_pairs = [[]]

    for layer in range(1, layers):
        temp_a_pair = np.hstack([feat_a[layer], feat_ap[layer]])
        a_pairs.append(np.vstack([temp_a_pair]))

        index_params[layer] = flann[layer].build_index(a_pairs[layer], algorithm='kdtree')

    return flann, index_params, a_pairs


def extract_pixel_feature(bp_l1, bp_l2, px, drop=False):
    # first extract full feature vector
    # since the images are padded, we need to add the padding to our indexing
    conf = Config()
    im_sm_padded, im_lg_padded = bp_l1, bp_l2
    row, col = px[0], px[1]
    px_feat = np.hstack([im_sm_padded[int(np.floor(row/2.)): int(np.floor(row/2.) + 2 * conf.pad_l1 + 1),
                                      int(np.floor(col/2.)): int(np.floor(col/2.) + 2 * conf.pad_l1 + 1)].flatten(),
                         im_lg_padded[row : int(row + 2 * conf.pad_l2 + 1),
                                      col : int(col + 2 * conf.pad_l2 + 1)].flatten()])

    if drop:
        return px_feat[:int(3 * ((conf.nl1 * conf.nl1) + conf.nfine))]
    else:
        return px_feat


def construct_feature_vector(single_pyramid):
    """

    :param single_pyramid:
    :return:
    """
    all_layer_feature_list = []
    # take every layer and append to comprehensive feature vector
    # use a single decomposed vector for the 5x5 kernel and map the layer to the vector.
    # take those single vectors and assign it to the one pixel we are trying to analyze
    # once we do that for all pixels, this should complete the feature vector for that one layer.
    # TODO: pad the images to handle edge cases
    for layer in single_pyramid:
        h, w = layer.shape[0:2]
        layer_feat = np.zeros([h, w, 25], dtype=np.float32)
        for i in range(h):
            for j in range(w):
                pixel_feat = np.zeros([25], dtype=np.float32)
                # if images are padded we dont have to skip the edge cases.
                if 2 <= i < h-2 and 2 <= j < w-2:
                    idx = 0
                    for x in range(-2,3):
                        for y in range(-2,3):
                            pixel_feat[idx] = layer[i+y, j+x]
                            idx += 1
                    layer_feat[i,j] = pixel_feat
        all_layer_feature_list.append(layer_feat)
    return all_layer_feature_list


def dimension_flat_for_ann(single_pyramid):
    """

    :param single_pyramid:
    :return:
    """
    # transform dimension in order to perform ANN
    h, w = single_pyramid.shape[0:2]
    reshaped_pyramid = kernel * np.array(
        np.reshape(single_pyramid, [h*w,25]),
        dtype=np.float32
    )
    return reshaped_pyramid
