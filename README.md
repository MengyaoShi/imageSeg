# imageSeg
Follow the following recipe:
git clone https://github.com/MengyaoShi/imageSeg.git
cd faster-rcnn.pytorch

you need to read readme in this folder, and follow pip installation and make requirements.

mkdir data
cd  data
mkdir science
cd science
mkdir train
mkdir test
cd  train
mkdir image

And then put our dataset in image folder.

Uder image/ folder, you should see

"00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552
003cee89357d9fe13516167fd67b609a164651b21934585648c740d2c3d86dc1
00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e
..."
in each of subdirectory like 00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552, there is our input image and several masks. Just keep original format.

outside image folder, under science/train, you should need to copy stage1_train_labels.csv over to current location.

so far you should have science/train/stage1_train_labels.csv science/train/image/...


go back to folder: faster-rcnn.pytorch
cd faster-rcnn.pytorch

mkdir save
cd save
mkdir science
mkdir vgg16

copy the vgg16 model from https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0 to vgg16 folder

so you should see save/science, save/vgg16/vgg16_caffe.pth

At this point you are all set!

go back to faster-rcnn.pytorch, run

python train_data_preprocess.py
Now you in faster-rcnn.pytorch/save/science you should see preprocessed input training set.

run:
python trainval_net1.py your data starts get training! 



