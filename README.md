# MRtoUS

Generate US images from MR brain images.

## Setup

1. Download the [group2 dataset][dataset].
2. Download and install the [minc toolset][toolset].

[dataset]: http://www.bic.mni.mcgill.ca/%7Elaurence/data/data.html
[toolset]: http://bic-mni.github.io

## Registration

Uses ground truth landmarks to align US and MR volumes.

```shell
# export the register transformation in the GUI
register -sync 01_mr_tal.mnc 01a_us_tal.mnc 01_all.tag

# apply the transformation to ultrasound images
mincresample 01a_us_tal.mnc 01_us_reg.mnc -transformation 01_reg.xfm -like 01_mr_tal.mnc
```

## Conversion

The MNI-BITE dataset uses [MINC-1.0][minc1] and [MINC-2.0][minc2] format. Later
extends [HDF5][hdf5] which has a maintained python implementation [h5py][h5py].

[h5py]: http://www.h5py.org
[hdf5]: https://en.wikipedia.org/wiki/Hierarchical_Data_Format
[minc1]: https://en.wikibooks.org/wiki/MINC/SoftwareDevelopment/MINC1_File_Format_Reference
[minc2]: https://en.wikibooks.org/wiki/MINC/SoftwareDevelopment/MINC2.0_File_Format_Reference

```shell
mincconvert -2 01_mr_tal.mnc 01_mr.mnc
```

## Dependencies

Install python dependencies.

```shell
pip3 install -r requirements.txt
```

## Evaluation

This will train a simple model with 30 epochs and then show you the results.

```shell
python3 scripts/evaluate.py --datadir=mnibite --train=13 --epochs=30
```

![Result1](https://cloud.githubusercontent.com/assets/1780466/23397311/4249227e-fd98-11e6-9de5-1b4e5213f2a1.png)
![Result2](https://cloud.githubusercontent.com/assets/1780466/23397313/44265e40-fd98-11e6-99c5-695585debeb8.png)