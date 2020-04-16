## Introduction
Comparative Test Unet & other variants

## Depence

Python 3.5.7

Tensorflow 1.12.0

Pillow 6.0.0

numpy 1.15.2

opencv_python 4.1.0


## Usage

To training a model and evaluate every epoch, create tfrecord for datas by using `DataProcessor/LoadArguandSavetif.py.py` , and put .tfrecord files in `/Data` . Run `python main.py`

To evaluate the trained model on the test set without ground-truth labels,  create tfrecord for datas by using `DataProcessor/LoadArguandSavetif.py.py` , and put .tfrecord files in `/Data` .  Run `python Main.py --Mode=Visualize --mod_dir=[path to model] --img_dir=[path to test images]`. 



## Explanations
/Data/:  Datas(Trainset.tfrecord  Testset.tfrecord)

/DataProcessor/:  Data processing tools
      LoadArguandSavetif.py: .tif to .tfrecord
      Processor.py :  Data processing functions( Augmentation function,  Image enhancement, Image Intensity, image splite&merge Image downsampling)

Readtf.py:  Method for read train/test set

/Nets/:  Network structure

Unet.py & Unet_Res.py& Unet_Res_Dia.py:  Comparative test networks
Layers.py & Layers_Dia.py:  Layers packages

/TestImage/  & /TestRes/:  Examples

Main.py:  Main function

TFUtils.py & TestUtils.py:   Train&Test Utils
