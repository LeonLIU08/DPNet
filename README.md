Keras for DPNet
===============
source code of DPNet paper

File structure
==============


Running
=======
### Prepare the feature map
    $ python feature_extraction.py 
             --test_video ./example/tr2.avi 
             --reference_video ./example/tr1.avi 
             --IQA SSIM 
             --patch_size 16 
             --save_file

Wait for the process finishing. Because of the optical flow operation is slow, it takes time...

The result will be stored in the "/tmp" folder.

### Quality prediction
    $ python video_test.py 
             --model ./model/LIVE_model_gmsd{32_8}.h5
             --feature_maps ./tmp/feaMap_tr2_GMSD.h5 
             --show_quality_changes
    
The overall quality score of the test video will be printed. The quality changes along with the time will also be provided, if the "--show_quality_changes" is chosen. 

