import numpy as np
from skimage.color import rgb2gray, rgb2yuv
from skimage.measure import compare_ssim as ssim
from pyflow import pyflow
from scipy import signal
import cv2

def gmsd(vref, vcmp, rescale=True, returnMap=False):
    """
    Compute Gradient Magnitude Similarity Deviation (GMSD) IQA metric
    :cite:`xue-2014-gradient`. This implementation is a translation of the
    reference Matlab implementation provided by the authors of
    :cite:`xue-2014-gradient`.
    Parameters
    ----------
    vref : array_like
      Reference image
    vcmp : array_like
      Comparison image
    rescale : bool, optional (default True)
      Rescale inputs so that `vref` has a maximum value of 255, as assumed
      by reference implementation
    returnMap : bool, optional (default False)
      Flag indicating whether quality map should be returned in addition to
      scalar score
    Returns
    -------
    score : float
      GMSD IQA metric
    quality_map : ndarray
      Quality map
    """

    # Input images in reference code on which this implementation is
    # based are assumed to be on range [0,...,255].
    if rescale:
        scl = (255.0/(vref.max()+0.000001))
    else:
        scl = np.float32(1.0)

    T = 170.0
    dwn = 2
    dx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])/3.0
    dy = dx.T

    ukrn = np.ones((2, 2))/4.0
    aveY1 = signal.convolve2d(scl*vref, ukrn, mode='same', boundary='symm')
    aveY2 = signal.convolve2d(scl*vcmp, ukrn, mode='same', boundary='symm')
    Y1 = aveY1[0::dwn, 0::dwn]
    Y2 = aveY2[0::dwn, 0::dwn]

    IxY1 = signal.convolve2d(Y1, dx, mode='same', boundary='symm')
    IyY1 = signal.convolve2d(Y1, dy, mode='same', boundary='symm')
    grdMap1 = np.sqrt(IxY1**2 + IyY1**2)

    IxY2 = signal.convolve2d(Y2, dx, mode='same', boundary='symm')
    IyY2 = signal.convolve2d(Y2, dy, mode='same', boundary='symm')
    grdMap2 = np.sqrt(IxY2**2 + IyY2**2)

    quality_map = (2*grdMap1*grdMap2 + T) / (grdMap1**2 + grdMap2**2 + T)
    score = np.std(quality_map)

    if returnMap:
        return (score, quality_map)
    else:
        return score


def GMSDqualitymap(img0, img1, patch_size, patch_stride):
    H, W, c = img1.shape
    output_h = (H - patch_size) / patch_stride + 1
    output_w = (W - patch_size) / patch_stride + 1
    output = np.zeros((output_h, output_w))

    for i in range((H - patch_size) / patch_stride + 1):
        for j in range((W - patch_size) / patch_stride + 1):
            patch0 = img0[i * patch_stride:i * patch_stride + patch_size,
                     j * patch_stride:j * patch_stride + patch_size, 0]
            patch1 = img1[i * patch_stride:i * patch_stride + patch_size,
                     j * patch_stride:j * patch_stride + patch_size, 0]
            q = gmsd(patch0, patch1)
            output[i, j] = q

    return output


def SSIMqualitymap(img0, img1, patch_size, patch_stride):
    H, W, c = img1.shape
    output_h = (H - patch_size) / patch_stride + 1
    output_w = (W - patch_size) / patch_stride + 1
    output = np.zeros((output_h, output_w))

    for i in range((H - patch_size) / patch_stride + 1):
        for j in range((W - patch_size) / patch_stride + 1):
            patch0 = img0[i * patch_stride:i * patch_stride + patch_size,
                     j * patch_stride:j * patch_stride + patch_size, 0]
            patch1 = img1[i * patch_stride:i * patch_stride + patch_size,
                     j * patch_stride:j * patch_stride + patch_size, 0]
            q = ssim(patch0, patch1, data_range=1.)
            output[i, j] = 1.-min(max(0., q), 1.)

    return output


def get_qualitymap(img0, img1, args):

    assert img0.shape[2] == 3, 'The input should be a RGB image.'
    assert img1.shape[2] == 3, 'The input should be a RGB image.'

    img0_yuv = rgb2yuv(img0)[:, :, 0:1]
    img1_yuv = rgb2yuv(img1)[:, :, 0:1]

    if args.IQA == 'GMSD':
        quailt_map = GMSDqualitymap(img0_yuv, img1_yuv,
                                    patch_size=args.patch_size, patch_stride=args.patch_stride)
    elif args.IQA == 'SSIM':
        quailt_map = SSIMqualitymap(img0_yuv, img1_yuv,
                                    patch_size=args.patch_size, patch_stride=args.patch_stride)
    else:
        quailt_map = 0
        assert False, "Currently, the system only support the SSIM and the GMSD as the FR-IQA method."

    return quailt_map


def OFmap(of_map, patch_size, patch_stride, modeltype='ave'):
    H, W = of_map.shape
    output_h = (H - patch_size) / patch_stride + 1
    output_w = (W - patch_size) / patch_stride + 1
    output = np.zeros((output_h, output_w))

    for i in range((H - patch_size) / patch_stride + 1):
        for j in range((W - patch_size) / patch_stride + 1):
            patch0 = of_map[i * patch_stride:i * patch_stride + patch_size,
                     j * patch_stride:j * patch_stride + patch_size]

            if modeltype == 'ave':
                output[i, j] = np.mean(patch0)
            else:
                output[i, j] = np.std(patch0)

    return output


def opticalflow(img1, img2):
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0

    if np.max(img1) > 1:
        img1 = img1 / 255.
        img2 = img2 / 255.

    u, v, im2w = pyflow.coarse2fine_flow(
        img1, img2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)

    return u, v


def of_extract(img1, img2, args):
    u, v = opticalflow(img1, img2)

    optiF_u = u.astype(np.float)
    optiF_v = v.astype(np.float)

    mag, agn = cv2.cartToPolar(optiF_u, optiF_v)

    of_ave_map = OFmap(mag,
                       patch_size=args.patch_size, patch_stride=args.patch_stride, modeltype='ave')
    of_std_map = OFmap(mag,
                       patch_size=args.patch_size, patch_stride=args.patch_stride, modeltype='std')

    return of_ave_map, of_std_map




