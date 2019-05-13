'''
This script takes the video input and output the feature maps.
The output feature maps can be stored in the /tmp.
'''

import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt
import skvideo.io
# import cv2
from utils.featureMaps import get_qualitymap, of_extract
from utils.tools import get_outputSize
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_video",
                        help="The path to the tested video.")
    parser.add_argument("--reference_video",
                        help="The path to the reference video.")
    parser.add_argument("--IQA", default='GMSD',
                        help="The IQA method to generate the quality map.")
    parser.add_argument("--patch_size", default=32, type=int,
                        help="Patch size.")
    parser.add_argument("--patch_stride", default=8, type=int,
                        help="Patch stride.")

    parser.add_argument('--save_file', action='store_true',
                        help="Save to file or not")

    args = parser.parse_args()

    return args


def main(args):
    tes_video = args.test_video
    ref_video = args.reference_video

    # get frame rate
    metadata = skvideo.io.ffprobe(tes_video)

    meta = str(metadata['video']["@avg_frame_rate"])
    frame_rate = int(meta.split('/')[0])/float(meta.split('/')[1])
    assert frame_rate >= 12, 'The frame rate of the input video should be higher than 12.'

    reader1 = skvideo.io.FFmpegReader(tes_video)
    reader2 = skvideo.io.FFmpegReader(ref_video)
    reader3 = skvideo.io.FFmpegReader(ref_video)

    # for the next frame
    for ref_frame in reader3.nextFrame():
        break

    Fnum, H, W, C = reader1.getShape()

    out_Fnum, out_H, out_W = get_outputSize(Fnum, H, W, args)
    print('The size of the output feature map will be:')
    print('-------------------------> \033[91m%d frames with %dx%d size \033[0m' %
          (out_Fnum, out_H, out_W))

    V_ave = np.zeros((out_Fnum, out_H, out_W))
    V_std = np.zeros((out_Fnum, out_H, out_W))
    M = np.zeros((out_Fnum, out_H, out_W))

    fidx = 0
    for tes_frame in reader1.nextFrame():
        print('\033[91m No.%d frame is processing... \033[0m' % (fidx+1))
        for ref_frame in reader2.nextFrame():
            break
        for ref_frame2 in reader3.nextFrame():
            break

        print('Calculate the quality map with %s' % (args.IQA))
        qualitymap = get_qualitymap(tes_frame, ref_frame, args)
        print('Calculate the motion map')
        ave, std = of_extract(ref_frame, ref_frame2, args)

        M[fidx, :, :] = qualitymap
        V_ave[fidx, :, :] = ave
        V_std[fidx, :, :] = std

        fidx += 1
        if fidx >= Fnum-1:
            break

    if args.save_file:
        file_name = './tmp/feaMap_%s_%s.h5' % (tes_video.split('/')[-1].split('.')[0], args.IQA)
        with h5py.File(file_name, 'w') as hf:
            hf.create_dataset('spatial', data=M)
            hf.create_dataset('ave', data=V_ave)
            hf.create_dataset('std', data=V_std)
            hf.create_dataset('frame_rate', data=frame_rate)

    return M, V_ave, V_std


if __name__ == '__main__':
    args = parse_args()
    main(args)
