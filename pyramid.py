import cv2
import numpy as np
import scipy.signal as sig


def create_arbitrary_pyramid(src_pyr):
    """
    Takes the filled image pyramid list data structure and builds arbitrary B prime
    layers and returns the output. The values are floats between 0. and 1.

    :param src_dict: gaussian pyramid data structure
    :param src_key: key of the source image you want to copy over (shape-wise)
    :param dst_key: key of the target iamge you want
    :return src_dict: the modified data structure that can be rewritten.
    """
    pyr_bp = []
    for layer in range(len(src_pyr)):
        pyr_bp.append(np.random.uniform(0, 1, size=src_pyr[layer].shape))

    return pyr_bp
    # src_dict[dst_key] = []
    # for layer in src_dict[src_key]:
    #     img_shape = layer.shape
    #     src_dict[dst_key].append(np.random.rand(np.product(img_shape)).reshape(img_shape))
    # return src_dict


def construct_pyramid(img_data, threshold):
    """
    Constructs the rest of the pyramid in all images image data provided as input.
    The old implementation for building pyramids is found in pyramid.py but for better
    performance, it was replaced with cv2.pyrDown.

    :param img_data: original image data of an image
    :param threshold: the smallest threshold value the image can go down until
    :return pyr_dict: updated dict with all the relevant gaussian pyramid layers in the list
    """
    pyr = [img_data]
    n_layers = 0
    img_size = min(img_data.shape[0:2])
    while img_size > threshold:
        n_layers += 1
        img_size = int(img_size/2)

    for _ in range(n_layers):
        curr_img = pyr[-1]
        down_img = cv2.pyrDown(curr_img)
        pyr.append(down_img)
    return pyr

    # # iterate over all input images, A, A_p, B
    # for image_type in pyr_dict:
    #     for i in range(n_layers):
    #         curr_img = pyr_dict[image_type][-1]
    #         down_img = cv2.pyrDown(curr_img)
    #         pyr_dict[image_type].append(down_img)
    #
    # return pyr_dict


def reverse_pyramid(pyr_dict):
    """

    :param pyr_dict:
    :return:
    """
    for image_type in pyr_dict:
        pyr_dict[image_type].reverse()
    return pyr_dict


def generatingKernel(a):
    """Return a 5x5 generating kernel based on an input parameter (i.e., a
    square "5-tap" filter.)

    Parameters
    ----------
    a : float
        The kernel generating parameter in the range [0, 1] used to generate a
        5-tap filter kernel.

    Returns
    -------
    output : numpy.ndarray
        A 5x5 array containing the generated kernel
    """
    # DO NOT CHANGE THE CODE IN THIS FUNCTION
    kernel = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
    return np.outer(kernel, kernel)


def reduce_layer(image, kernel=generatingKernel(0.4)):
    """Convolve the input image with a generating kernel and then reduce its
    width and height each by a factor of two.

    For grading purposes, it is important that you use a reflected border
    (i.e., padding equivalent to cv2.BORDER_REFLECT101) and only keep the valid
    region (i.e., the convolution operation should return an image of the same
    shape as the input) for the convolution. Subsampling must include the first
    row and column, skip the second, etc.

    Example (assuming 3-tap filter and 1-pixel padding; 5-tap is analogous):

                          fefghg
        abcd     Pad      babcdc   Convolve   ZYXW   Subsample   ZX
        efgh   ------->   fefghg   -------->  VUTS   -------->   RP
        ijkl    BORDER    jijklk     keep     RQPO               JH
        mnop   REFLECT    nmnopo     valid    NMLK
        qrst              rqrsts              JIHG
                          nmnopo

    A "3-tap" filter means a 3-element kernel; a "5-tap" filter has 5 elements.
    Please consult the lectures for a more in-depth discussion of how to
    tackle the reduce function.

    Parameters
    ----------
    image : numpy.ndarray
        A grayscale image of shape (r, c). The array may have any data type
        (e.g., np.uint8, np.float64, etc.)

    kernel : numpy.ndarray (Optional)
        A kernel of shape (N, N). The array may have any data type (e.g.,
        np.uint8, np.float64, etc.)

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        An image of shape (ceil(r/2), ceil(c/2)). For instance, if the input is
        5x7, the output will be 3x4.
    """
    # use the 5 tap with the kernel defined above.
    # Convolve with the kernel and downsample every other with ::2 indexing
    # use sci-py's convolve function
    image = cv2.copyMakeBorder(image, 2, 2, 2, 2, cv2.BORDER_REFLECT_101)
    node = sig.convolve2d(image, kernel, 'valid')
    return node[::2, ::2]


def gaussPyramid(image, levels):
    """Construct a pyramid from the image by reducing it by the number of
    levels specified by the input.

    You must use your reduce_layer() function to generate the pyramid.

    Parameters
    ----------
    image : numpy.ndarray
        An image of dimension (r, c).

    levels : int
        A positive integer that specifies the number of reductions to perform.
        For example, levels=0 should return a list containing just the input
        image; levels = 1 should perform one reduction and return a list with
        two images. In general, len(output) = levels + 1.

    Returns
    -------
    list<numpy.ndarray(dtype=np.float)>
        A list of arrays of dtype np.float. The first element of the list
        (output[0]) is layer 0 of the pyramid (the image itself). output[1] is
        layer 1 of the pyramid (image reduced once), etc.
    """

    # iterate through the levels specified and recurse on the image to produce the nodes on the pyramid
    # actually just use a temp array since the prototype isnt meant for recursion.
    pyramid = [image.astype(np.float64)]
    for i in range(levels):
        # use the default kernel 0.4 for gaussian - 0.5 for pyramid
        image = reduce_layer(image)
        pyramid.append(image.astype(np.float64))
    return pyramid
