# Perception Challenge for Bin Picking

[![build](https://github.com/Yadunund/ibpc/actions/workflows/build.yaml/badge.svg?branch=main)](https://github.com/Yadunund/ibpc/actions/workflows/build.yaml)
[![style](https://github.com/Yadunund/ibpc/actions/workflows/style.yaml/badge.svg?branch=main)](https://github.com/Yadunund/ibpc/actions/workflows/style.yaml)

For more details on the challenge, [click here](https://bpc.opencv.org/).

## Design

TODO

## Requirements
- [Docker](https://docs.docker.com/)

> Note: Participants are expected to submit Docker containers, so all development workflows are designed with this in mind.

## Setup

```bash
mkdir ~/ws_ibpc/src -p
cd ~/ws_ibpc/src
git clone https://github.com/Yadunund/ibpc.git
```

## Build

Build the `ibpc_pose_estimator`.

```bash
cd ~/ws_ibpc/src/ibpc
docker buildx build -t ibpc:pose_estimator \
    --file ./Dockerfile.estimator \
    --build-arg="MODEL_DIR=models" \
    .
```

Build the `ibpc_tester`.

```bash
cd ~/ws_ibpc/src/ibpc
docker buildx build -t ibpc:tester \
    --file ./Dockerfile.tester \
    --build-arg="BOP_PATH=datasets" \
    --build-arg="DATASET_NAME=lm" \
    .
```

## Run

### Start the Zenoh router
```bash
docker run --init --rm --net host eclipse/zenoh:1.1.1 --no-multicast-scouting
```

### Run the pose estimator

```bash
docker run --network=host ibpc:pose_estimator
```

### Run the tester

> Note: The BOP_PATH envar should point to a folder that contains models in the BOP format.
See https://bop.felk.cvut.cz/datasets/ for more details.

```bash
docker run --network=host -e BOP_PATH=/opt/ros/underlay/install/datasets -v/home/tullyfoote/ws/ibpc/lm:/opt/ros/underlay/install/datasets/lm -it ibpc:tester 

```
