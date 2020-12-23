import itertools

import numpy as np

from config import Config


# def create_flann_index(train):
#     params = dict(algorithm=1, trees=4)
#     flann_idx = cv2.flann_Index(train, params)
#     return flann_idx
#
#
# def flann_alg(ann_idx, layer, reduced_b_feat, reg_a_p_feat, pyramid_b_single_shape):
#     feat_b = np.array(reduced_b_feat[layer], dtype=np.float32)
#
#     idx, dst = ann_idx.knnSearch(feat_b, 1, params={})
#
#     col, row = idx % reg_a_p_feat.shape[1], idx / reg_a_p_feat.shape[1]
#
#     h, w = pyramid_b_single_shape[0:2]
#     match = np.zeros([h, w], dtype=np.float32)
#     for i in range(h):
#         for j in range(w):
#             if 2 <= i < h-2 and 2 <= j < w-2:
#                 k = i * w + j
#
#     row_min = row[k,0]
#     col_min = col[k,0]
#
#     match[i, j] = reg_a_p_feat[layer][row_min, col_min][12]
#
#     return row, col, match

def pixel_to_index(pxs, w):
    rows, cols = pxs[0], pxs[1]
    return (rows * w + cols).astype(int)


def index_to_pixel(ixs, w):
    cols = ixs % w
    rows = (ixs - cols) // w
    return np.array([rows, cols])


def Ap_index_to_pixel(ixs, h, w):
    pxs = index_to_pixel(ixs, w)
    rows, cols = pxs[0], pxs[1]
    img_nums = (np.floor(rows/h)).astype(int)
    img_ixs = ixs - img_nums * h * w
    return index_to_pixel(img_ixs, w), img_nums


def Ap_pixel_to_index(pxs, img_nums, h, w):
    rows, cols = pxs[0], pxs[1]
    return (((h * img_nums) + rows) * w + cols).astype(int)


def best_approximate_match(flann, params, bp_feat):
    result, dists = flann.nn_index(bp_feat, 1, checks=params['checks'])
    return result[0]


def best_coherence_match(As, h, w, bp_feat, s, im, px, Bp_w):
    conf = Config()
    row, col = px

    rs = []
    ims = []
    prs = []
    rows = np.arange(np.max([0, row - conf.pad_l2]), row + 1, dtype=np.int8)
    cols = np.arange(np.max([0, col - conf.pad_l2]), np.min([Bp_w, col + conf.pad_l2 + 1]), dtype=np.int8)

    for r_coord in itertools.product(rows, cols):
        if pixel_to_index(r_coord, Bp_w) < pixel_to_index(px, Bp_w):

            pr = s[pixel_to_index(r_coord, Bp_w)] + px - r_coord

            img_nums = im[pixel_to_index(r_coord, Bp_w)]

            if 0 <= pr[0] < h and 0 <= pr[1] < w:
                # keep the bound limited to original image pair
                rs.append(np.array(r_coord))
                ims.append(img_nums)
                prs.append(Ap_pixel_to_index(pr, img_nums, h, w))

    idx = np.argmin(
        np.linalg.norm(As[np.array(prs)] - bp_feat,
                       ord=2,
                       axis=1)
    )
    r_star = rs[idx]
    return s[pixel_to_index(r_star, Bp_w)] + px - r_star, 0, r_star


def compute_distance(A, B, weights):
    return np.linalg.norm((A - B) * weights, ord=2) ** 2
