# Industrial Bin Picking Challenge (IBPC)

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
rosdep update
rosdep install --from-paths src --ignore-src --rosdistro jazzy -y
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
```

### With Docker
```bash
cd ~/ws_ibpc/src/ibpc
docker buildx build -t ibpc:pose_estimator --file ./Dockerfile.estimator .
```

## Run the pose estimator

### On Ubuntu 24.04
```bash
cd ~/ws_ibpc/
source install/setup.bash
ros2 run ibpc_pose_estimator ibpc_pose_estimator --ros-args -p model_path:=<PATH>
```

### With Docker
```bash
docker run --network=host ibpc:pose_estimator -e MODEL_PATH=<PATH>
```

## Query the pose estimator
```bash
cd ~/ws_ibpc/
source install/setup.bash
ros2 service call /get_pose_estimates ibpc_interfaces/srv/GetPoseEstimates '{}'
```