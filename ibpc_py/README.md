# Industrial Bin Picking Challenge (IBPC)

This is the python entrypoint for the Industrial Bin Picking Challenge

## Usage

`bpc test <Pose Estimator Image Name> ~/<Path to dataset to test>`

`bpc test ibpc:pose_estimator ~/ws/ibpc/lm`


## Prerequisites


### Install the package:

In a virtualenv
`pip install ibpc`

Temporary before rocker release of https://github.com/osrf/rocker/pull/317/ 
`pip uninstall rocker && pip install git+http://github.com/osrf/rocker.git@console_to_file`


### Nvidia Docker (optoinal)
Make sure nvidia_docker is installed if you want cuda. 

