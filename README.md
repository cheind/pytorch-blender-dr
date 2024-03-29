# pytorch-blender-dr

Provides code to recreate our results on 2D object detection und keypoint detection using adaptive domain randomization. In particular, we show that training on massively randomized low-fidelity simulation generalizes to real world data.

## Two industrial use cases

### TLess - TextureLess Objects

Links
 - [TLess details](https://bop.felk.cvut.cz/datasets/#T-LESS)
 - [TLess format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md)
 - [Other challenges](https://bop.felk.cvut.cz/challenges/)
 
### Kitchen cabinets

TODO

## Generating data
Requires [blendtorch](https://github.com/cheind/pytorch-blender) + Blender 2.9  installed. For occluder objects we use [this supershape](https://github.com/cheind/supershape) library, that needs to be accessible from within Blender.

To generate data, run 
```
python record.py <scene>
``` 
where `<scene>` is `less` or `kitchen`. This will generate offline data (`.btr` files) in `tmp` directory. Example below.

<p align="center">
  <img src="etc/tless.jpg" width="500">
</p>

Invoke
```
python record.py --help
``` 
for more options.

## Training and Evaluation

Create a new Conda environment
```
conda create -n icpr python==3.7 pip
conda activate icpr
pip install -r requirements.txt
```

Setup BlendTorch:
```
Detailed instructions can be found at:
https://github.com/cheind/pytorch-blender
under 'Installation' in the readme.

git clone https://github.com/cheind/pytorch-blender.git <DST>
pip install -e <DST>/pkg_pytorch
```

How to run scripts:
```
Run the main script for training with:
>> python -m py_torch.main --config example_train.txt
Run the main script for mean average precision calculation with:
>> python -m py_torch.main --config example_test.txt
NOTE: Change train_path and inference_path in the example_train.txt
and example_test.txt s.t. it matches your data location!

The example configuration above will replay from existing 
BlendTorch replay file(s) with a '.btr' extension. Thus, run
the record.py script beforehand with:
>> python record.py tless
as stated above.

We train and choose the best performing model only by training
on BlendTorch generated data (~50k images) and evaluate the performance (mAP metric)
with real world data taken by a primesense camera (~1000 images).

During training models are stored in the 'models' folder
as model_last.pth or model_best.pth!

Train and evaluate multiple times with the 'ntimes.sh' script. It will use the
settings from example_train.txt and example_test.txt for training and evaluation
respectively.

Get detailed mAP summary report over single or multiple trainings runs (ntimes.sh)
by running the 'coco_report.py' file:
>> python -m py_torch.coco_report './evaluation'
where './evaluation' is the path to gt.json and pred.json (or multiple pred_*.json) file.
```
Track/Visualize training progress with tensorboard:
```
tensorboard --logdir runs --bind_all
```

Augmented BlendTorch image with ground truth and prediction annotations: 
<p align="center">
  <img src="etc/tensorboard/training_gt.png" width="250">
  <img src="etc/tensorboard/training.png" width="250">
</p>

Precision/Recall curves over single classes:
<p align="center">
  <img src="etc/tensorboard/pr_curves.png" width="500">
</p>

CenterNet's dense prediction output heatmaps (prediction, ground truth):
<p align="center">
  <img src="etc/tensorboard/heatmaps.png" width="500">
</p>
