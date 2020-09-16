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

## Training

Create a new Conda environment
```
conda create -n icpr python==3.7 pip
conda activate icpr
pip install -r requirements.txt
```

## Evaluation
TODO