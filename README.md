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

## Patching

To exclude empty training data (volume areas where we have no US available)
and still use batching we create overlapping 30x30 patches and filter patches.

```shell
python3 -m scripts.patch
```

## Evaluation

This will train a simple model with 30 epochs and show you live udpates
on loss and rendered images.

```shell
python3 -W ignore main.py --show-loss --show-images --show-patches --train 12 13
```

![images](https://cloud.githubusercontent.com/assets/1780466/23622246/3840871c-029e-11e7-840e-bab8f55a4c0f.png)
![patches](https://cloud.githubusercontent.com/assets/1780466/23622248/3a23ff96-029e-11e7-8d3a-6cba2238a966.png)