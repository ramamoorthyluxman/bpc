# Industrial Bin Picking Challenge (IBPC)

[![build](https://github.com/Yadunund/ibpc/actions/workflows/build.yaml/badge.svg?branch=main)](https://github.com/Yadunund/ibpc/actions/workflows/build.yaml)
[![style](https://github.com/Yadunund/ibpc/actions/workflows/style.yaml/badge.svg?branch=main)](https://github.com/Yadunund/ibpc/actions/workflows/style.yaml)

For more details on the challenge, [click here](https://bpc.opencv.org/).

## Design

TODO

## Requirements
- [Ubuntu 24.04](https://ubuntu.com/blog/tag/ubuntu-24-04-lts)
- [ROS 2 Jazzy Jalisco](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)
- [Optional] [Docker](https://docs.docker.com/)

> Note: Participants can rely on Docker images if they are unable to setup a native Ubuntu 24.04 environemnt with ROS 2 Jazzy Jalisco.

## Setup

```bash
mkdir ~/ws_ibpc/src -p
cd ~/ws_ibpc/src
git clone https://github.com/Yadunund/ibpc.git
```

## Build

### On Ubuntu 24.04
```bash
cd ~/ws_ibpc/
sudo apt update && sudo apt install ros-jazzy-rmw-zenoh-cpp
rosdep update
rosdep install --from-paths src --ignore-src --rosdistro jazzy -y
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
```

### With Docker
```bash
cd ~/ws_ibpc/src/ibpc
docker buildx build -t ibpc:pose_estimator \
    --file ./Dockerfile.estimator \
    --build-arg="MODEL_DIR=models" \
    . 
```

## Start the Zenoh router
```bash
docker run --init --rm --net host eclipse/zenoh:1.1.1 --no-multicast-scouting
```

## Run the pose estimator

### On Ubuntu 24.04
```bash
cd ~/ws_ibpc/
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
ros2 run ibpc_pose_estimator ibpc_pose_estimator --ros-args -p model_dir:=<PATH>
```

### With Docker
```bash
docker run --network=host ibpc:pose_estimator
```

## Query the pose estimator
```bash
cd ~/ws_ibpc/
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
ros2 service call /get_pose_estimates ibpc_interfaces/srv/GetPoseEstimates '{}'
```