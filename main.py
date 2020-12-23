import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from config import Config
from features import feature_vector, pad_img, kernel_lg, kernel_sm, ann_index, extract_pixel_feature
from images import Image
from match import best_approximate_match, best_coherence_match, compute_distance, pixel_to_index,\
    Ap_index_to_pixel, Ap_pixel_to_index
from pyramid import create_arbitrary_pyramid, construct_pyramid
# Using cv2 builtin pyr function instead.
# from pyramid import gaussPyramid

images_path = Path('images') / 'input'
output_path = Path('images') / 'output'

PYR_LEVELS = 5


def discover_image_dirs():
    return list(images_path.glob('*'))


def get_images_in_dir(posix_path):
    png = list(posix_path.glob('*.png'))
    jpg = list(posix_path.glob('*.jpg'))
    if png or jpg:
        return png + jpg
    else:
        logging.error('No image files discovered in the directories provided')
        raise Exception()


def create_img_obj(path):
    # have the filenames be A, A_p, B?
    # write this out.
    return Image(path)


def stacked_gaussian(config, kernel_lg, kernel_sm):
    reduced_kernel_sm = kernel_sm.flatten()
    reduced_kernel_lg = kernel_lg.flatten()
    stacked_gauss_sm = np.array([val for val in reduced_kernel_sm for _ in range(3)])
    stacked_gauss_lg = np.array([val for val in reduced_kernel_lg for _ in range(3)])

    weight_sm = 1. / (config.nl1 ** 2) * stacked_gauss_sm
    weight_lg = 1. / (config.nl2 ** 2) * stacked_gauss_lg
    weight_fine = 1. / (config.nfine) * stacked_gauss_lg[:int(config.nfine * 3)]

    return np.hstack([weight_sm, weight_lg, weight_sm, weight_fine])


def main():
    conf = Config()

    image_dirs = discover_image_dirs()
    images = []
    for image_dir in image_dirs:
        images.append(get_images_in_dir(image_dir))
    temp_images = []
    for image_dir in images:
        temp_images.append([create_img_obj(img_path) for img_path in image_dir])
    images = temp_images[0]
    # still need to load the image and add the metadata.
    for image in images:
        image.load_image_values()
        image.define_metadata()
        image.convert_rgb_to_yiq()

    for image in images:
        if image.metadata == 'A':
            image.convert_bgr_to_rgb()
            image.normalize_img_to_float()
            A_img = image
        elif image.metadata == 'B':
            image.convert_bgr_to_rgb()
            image.normalize_img_to_float()
            B_img = image
        elif image.metadata == 'A_p':
            image.convert_bgr_to_rgb()
            image.normalize_img_to_float()
            Ap_img = image
        else:
            print('Image with wrong format detected. Ignoring.')

    # Just use the Y channel for building pyr and features.
    if conf.use_yiq:
        A = A_img.yiq[:, :, 0]
        B = B_img.yiq[:, :, 0]
        A_p = Ap_img.yiq[:, :, 0]
    else:
        A = A_img.pixels
        B = B_img.pixels
        A_p = Ap_img.pixels

    # transform images to pyramid
    pyr_A = construct_pyramid(A, 3)
    pyr_A.reverse()
    pyr_B = construct_pyramid(B, 3)
    pyr_B.reverse()
    pyr_A_p = construct_pyramid(A_p, 3)
    pyr_A_p.reverse()
    pyr_B_p = create_arbitrary_pyramid(pyr_B)

    reference_pyr = [construct_pyramid(A_p, 3)]
    reference_pyr[0].reverse()

    print(f"pyramids built")
    # # initialize dict comprehension
    # image_pyramids = {img.metadata: [img.yiq] for img in images}
    # a_height, a_width = image_pyramids['A'][0].shape
    # b_height, b_width = image_pyramids['B'][1].shape
    #
    # image_pyramids = construct_pyramid(image_pyramids, 4)
    # image_pyramids = reverse_pyramid(image_pyramids)
    #
    # # create an arbitrary B prime pyramid layer
    # image_pyramids = create_arbitrary_pyramid(image_pyramids, 'B', 'B_p')

    # compute feature vector
    feat_B = feature_vector(pyr_B)
    print(f"feature vector for B built")

    # index creation for ANN
    flann, index_param, a_pair = ann_index(pyr_A, pyr_A_p)
    weights = stacked_gaussian(conf, kernel_lg, kernel_sm)
    print(f"ANN indices initialized")

    for layer in range(1, len(pyr_B)):
        print(f"working on layer {layer}")
        h, w = pyr_B_p[layer].shape[0], pyr_B_p[layer].shape[1]
        output_image = np.nan * np.ones((h, w, 3))

        s = []
        im = []

        for r in range(h):
            for c in range(w):
                px = np.array([r, c])

                # we need to use both the layer prior and the current layer
                bp_l1 = pad_img(pyr_B_p[layer - 1], conf.padding_l1)
                bp_l2 = pad_img(pyr_B_p[layer], conf.padding_l2)
                # create a holistic feature vector holding all this information
                bp_feat = np.hstack([feat_B[layer][pixel_to_index(px, w), :],
                                    extract_pixel_feature(bp_l1, bp_l2, px, drop=True)])

                # use the feature vector along with the ANN index we created with A and A_p above
                nn_idx = best_approximate_match(flann[layer], index_param[layer], bp_feat)

                Ap_h, Ap_w = pyr_A_p[layer].shape[0], pyr_A_p[layer].shape[1]
                pixel_approx, i_app = Ap_index_to_pixel(nn_idx, Ap_h, Ap_w)

                if len(s) < 1:
                    p = pixel_approx
                    i = i_app
                else:
                    # Find Coherence Match and Compare Distances
                    pixel_coher, i_coh, r_star = best_coherence_match(a_pair[layer], Ap_h, Ap_w, bp_feat, s, im, px, w)

                    if np.allclose(pixel_coher, np.array([-1, -1])):
                        p = pixel_approx

                    else:
                        A_pair_feat_app = a_pair[layer][nn_idx]
                        A_pair_feat_coh = a_pair[layer][Ap_pixel_to_index(pixel_coher, i_coh, Ap_h, Ap_w)]

                        dist_app = compute_distance(A_pair_feat_app, bp_feat, weights)
                        dist_coh = compute_distance(A_pair_feat_coh, bp_feat, weights)

                        if dist_coh <= dist_app * (1 + (2**(layer - len(pyr_B))) * conf.k):
                            p = pixel_coher
                            i = i_coh
                        else:
                            p = pixel_approx
                            i = i_app

                pyr_B_p[layer][r, c] = pyr_A_p[layer][tuple(p)]

                try:
                    output_image[r, c, :] = reference_pyr[0][layer][tuple(p)]
                except Exception as e:
                    print(e)

                s.append(p)
                im.append(i)

        plt.imsave(f"{output_path}/layer{layer}.jpg", output_image)
        print(f"layer {layer} complete.")


    #
    # for image_type in image_pyramids:
    #     if image_type == 'A':
    #         for i in range(len(image_pyramids['A'])):
    #             feature_A = construct_feature_vector(image_pyramids['A'])
    #     elif image_type == 'A_p':
    #         for i in range(len(image_pyramids['A_p'])):
    #             feature_A_p = construct_feature_vector(image_pyramids['A_p'])
    #     elif image_type == 'B':
    #         for i in range(len(image_pyramids['B'])):
    #             feature_B = construct_feature_vector(image_pyramids['B'])
    #     else:
    #         print('Image file in wrong format detected')
    #         raise Exception()
    #
    # red_A = []
    # red_B = []
    # for i in range(PYR_LEVELS):
    #     red_A.append(dimension_flat_for_ann(feature_A[i]))
    #     red_B.append(dimension_flat_for_ann(feature_B[i]))
    #
    # layer = 0
    # train_data = np.array(red_A[layer], dtype=np.float32)
    #
    # flann_idx = create_flann_index(train_data)
    #
    # rlist, clist, B_p = flann_alg(flann_idx, layer, red_B,
    #                               feature_A_p, image_pyramids['B'][layer].shape)

    # # perform best match for every level for every pixel.
    # for level in range(0, PYR_LEVELS):
    #     # for row for cols
    #     pass


if __name__ == "__main__":
    main()
