import h5py
import numpy as np
import argparse
from keras.models import load_model, Model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--feature_maps', default='tmp/feaMap_tr2_GMSD.h5')
    parser.add_argument('--model', default='model/LIVE_model_gmsd{32_8}.h5',
                        help="The path the trained model file.")

    parser.add_argument('--show_quality_changes', action='store_true',
                        help="Save to file or not")

    args = parser.parse_args()

    return args


def build_fragment(featuremap, frame_rate):
    # the target output featuremap is 10s long, 12.5 frame rate

    def __stride_calculation(length, frg_num):
        if frg_num > 1:
            stride = int((length-10)/(frg_num-1))
        else:
            stride = 1
        return stride

    def __length_calculation(frame_num, frq_frg):
        length = int(frame_num/frq_frg)
        return length

    frame_num, H, W = featuremap.shape
    video_len = int(frame_num/frame_rate)

    assert video_len >= 9, 'The input video should longer than 9 second'

    time_frg = int(video_len / 10) + 1
    frq_frg = int(round(frame_rate/12.5))
    num_frg = time_frg*frq_frg

    output = np.zeros((num_frg, 125, H, W, 1))

    time_stride = __stride_calculation(video_len, time_frg)
    frq_len = __length_calculation(frame_num, frq_frg)
    frgs = []
    for i in range(time_frg):
        time_frgs = featuremap[i*frame_rate*time_stride:(i*time_stride+10)*frame_rate]
        for j in range(frq_frg):
            a = time_frgs[j::frq_frg, :, :]
            l = a.shape[0]
            output[i*frq_frg+j, :l, :, :, 0] = a

    # print(output.shape)
    return output


def main(args):

    # load feature maps file
    with h5py.File(args.feature_maps, 'r') as hf:
        spatial = hf['spatial'][:]
        ave = hf['ave'][:]
        std = hf['std'][:]
        # frame_rate = hf['frame_rate'].value
    frame_rate = 25

    # split it to 10s, 12.5 fps fragment
    frame_num = spatial.shape[0]
    video_len = int(frame_num/frame_rate)

    spa_array = build_fragment(spatial, frame_rate)
    ave_array = build_fragment(ave, frame_rate)
    std_array = build_fragment(std, frame_rate)

    # load the model
    model = load_model(args.model)
    # print(model.summary())
    input_shape = model.get_layer('input_spa').output_shape

    if input_shape[2] < spa_array.shape[2] or input_shape[3] < spa_array.shape[3]:
        # crop the feature map to desire size
        print('The dimension of the input feature map is not fit the model input.')
        print('The feature map will be automatically cropped to the required dimension.')
        print('The performance cannot be guaranteed.')
        spa_array = spa_array[:, :, :input_shape[2], :input_shape[3], :]
        ave_array = ave_array[:, :, :input_shape[2], :input_shape[3], :]
        std_array = std_array[:, :, :input_shape[2], :input_shape[3], :]
    elif input_shape[2] > spa_array.shape[2] or input_shape[3] > spa_array.shape[3]:
        # crop the feature map to desire size
        print('The dimension of the input feature map is not fit the model input.')
        print('The feature map will be automatically cropped to the required dimension.')
        print('The performance cannot be guaranteed.')
        spa_array2 = np.zeros((spa_array.shape[0], 125, input_shape[2], input_shape[3], 1))
        spa_array2[:, :, :spa_array.shape[2], :spa_array.shape[3], :] = spa_array
        ave_array2 = np.zeros((ave_array.shape[0], 125, input_shape[2], input_shape[3], 1))
        ave_array2[:, :, :ave_array.shape[2], :ave_array.shape[3], :] = ave_array
        std_array2 = np.zeros((std_array.shape[0], 125, input_shape[2], input_shape[3], 1))
        std_array2[:, :, :std_array.shape[2], :std_array.shape[3], :] = std_array

        spa_array = spa_array2
        ave_array = ave_array2
        std_array = std_array2

    print('%d video fragments are generated. The final score is the average value of them.' %
          (spa_array.shape[0]))
    # print(spa_array.shape)
    # put the input to the model
    output = model.predict([spa_array, ave_array, std_array])
    # print(output)
    if args.show_quality_changes:
        layer_name = 'td_dense1'
        middlelayer_model = Model(inputs=model.input,
                                  outputs=model.get_layer(layer_name).output)
        time_seq = middlelayer_model.predict([spa_array, ave_array, std_array])

        print('The quality change along with time:')
        print(np.abs(np.mean(time_seq, axis=0)[:, 0]))

    print('Overall Quality score is : %.4f/100.' % np.mean(output))
    print('Higher score means worse quality.')


if __name__ == '__main__':
    args = parse_args()
    main(args)