import numpy as np


def get_outputSize(Fnum, H, W, args):

    outH = (H-args.patch_size)/args.patch_stride + 1
    outW = (W-args.patch_size)/args.patch_stride + 1

    return Fnum-1, outH, outW
