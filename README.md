# SfMLearner
This codebase implements the system described in the paper for MPI sintel:

Unsupervised Learning of Depth and Ego-Motion from Video

[Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz/), [Matthew Brown](http://matthewalunbrown.com/research/research.html), [Noah Snavely](http://www.cs.cornell.edu/~snavely/), [David G. Lowe](http://www.cs.ubc.ca/~lowe/home.html)

In CVPR 2017 (**Oral**).

See the [project webpage](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/) 


## Prerequisites
This codebase was developed and tested with Tensorflow 1.0, CUDA 8.0 and Ubuntu 16.04 and is compatable with windows OS.

## Preparing training data
In order to train the model using the provided code, the data needs to be formatted in a certain manner. 

For [MPIsintel](http://sintel.is.tue.mpg.de/downloads), first download the dataset complete dataset provided on the official website, and then run the following command. Code for this part is available [here](https://github.com/RamSrivatsav/MPIdataprep).
```bash
python MPIDataPrep/prepare_train_data.py --dataset_dir MPISintel/MPI-Sintel-complete/training/clean --cam_dir MPISintel/MPI-Sintel-depth-training-20150305/training/camdata_left --dataset_name MPI_Sintel --dump_root MPIDataPrep/formatted/data --seq_length=3 --img_width=416 --img_height=128 --num_threads=4
```

## Training
Once the data are formatted following the above instructions, you should be able to train the model by running the following command
```bash
python train.py --dataset_dir=/path/to/the/formatted/data/ --checkpoint_dir=/where/to/store/checkpoints/ --img_width=416 --img_height=128 --batch_size=4
```
You can then start a `tensorboard` session by
```bash
tensorboard --logdir=/path/to/tensorflow/log/files --port=8888
```
and visualize the training progress by opening [https://localhost:8888](https://localhost:8888) on your browser. If everything is set up properly, you should start seeing reasonable depth prediction after ~100K iterations when training on MPIsintel. 

## MPIsintel Testing code

### Depth
Once you have model trained, you can obtain the single-view depth predictions on the MPIsintel eigen test split formatted properly for evaluation by running
```bash
python test_kitti_depth.py --dataset_dir /path/to/raw/MPIsintel/dataset/ --output_dir /path/to/output/directory --ckpt_file /path/to/pre-trained/model/file/
```
Again, a sample model can be downloaded by
```bash
bash ./models/download_depth_model.sh
```

### Pose
We also provide sample testing code for obtaining pose predictions on the MPIsintel dataset with a pre-trained model. You can obtain the predictions formatted as above for pose evaluation by running
```bash
python test_kitti_pose.py --test_seq [sequence_id] --dataset_dir /path/to/MPIsintel/odometry/set/ --output_dir /path/to/output/directory/ --ckpt_file /path/to/pre-trained/model/file/
```
A sample model trained on 5-frame snippets can be downloaded by
```bash
bash ./models/download_pose_model.sh
```
Then you can obtain predictions on, say `Seq. 9`, by running
```bash
python test_kitti_pose.py --test_seq 9 --dataset_dir /path/to/MPIsintel/odometry/set/ --output_dir /path/to/output/directory/ --ckpt_file models/model-100280
```

## Other implementations
[Pytorch](https://github.com/ClementPinard/SfmLearner-Pytorch) (by Clement Pinard)

## Disclaimer
This is an unofficial implementation of the system described in the paper and not an official Google product.
