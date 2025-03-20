# Troubleshooting errors

## Docker permission error

### Problem

```
permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Get "http://%2Fvar%2Frun%2Fdocker.sock/v1.47/containers/json": dial unix /var/run/docker.sock: connect: permission denied
```
### Solution

```
sudo chmod 666 /var/run/docker.sock
```

## Docker gpu detection problem

### Problem

```
docker run --rm --name bpc_estimator  --network host   --gpus all bdac47bec806 
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]

Run 'docker run --help' for more information
Non-interactive Docker run failed
 Command '['docker', 'run', '--rm', '--name', 'bpc_estimator', '--network', 'host', '--gpus', 'all', 'bdac47bec806']' returned non-zero exit status 125.
Estimator finished with exit code 125
Non-interactive Docker run failed
 Command '['docker', 'run', '--rm', '--name', 'bpc_zenoh', '--network', 'host', '86a36af59cff']' returned non-zero exit status 137.
bpc_zenoh finished with exit code 137 -- stopping others.
Non-interactive Docker run failed
 Command '['docker', 'run', '--rm', '-e', 'BOP_PATH=/opt/ros/underlay/install/datasets/', '-e', 'DATASET_NAME=ipd', '--name', 'bpc_tester', '--network', 'host', '-v', '/home/ram:/opt/ros/underlay/install/datasets', '-v', '/home/ram:/submission', '23f899a57e08']' returned non-zero exit status 137.
bpc_tester finished with exit code 137 -- stopping others.
```

### Solution

#### Add the NVIDIA container repository
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```

#### Update and install
```
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

#### Restart docker
```
sudo systemctl restart docker
```

## Explanation on BPC Container System Architecture

### Overview

The Bin Picking Challenge (BPC) framework uses a 3 Docker containers that work together to evaluate pose estimation algorithms. These containers are defined at the beginning of the code ibpc.py (bpc python library):

```python
ESTIMATOR_CONTAINER = "bpc_estimator"
TESTER_CONTAINER = "bpc_tester"
ZENOH_CONTAINER = "bpc_zenoh"
DEFAULT_CONTAINER_NAMES = [ESTIMATOR_CONTAINER, ZENOH_CONTAINER, TESTER_CONTAINER]
```

### ESTIMATOR_CONTAINER ("bpc_estimator")

This is the primary container that implements the pose estimation algorithm:

- **Purpose**: Processes input data to identify objects and estimate their 6D poses (position and orientation)
- **Key Features**:
  - Deep learning models for object detection and pose estimation
  - GPU acceleration (requires NVIDIA support)
  - Exposes service endpoint (`/get_pose_estimates`)
- **Default Image**: `ghcr.io/opencv/bpc/bpc_pose_estimator:example`
- **Customization**: Users can supply their own implementation as a command-line argument

### TESTER_CONTAINER ("bpc_tester")

The evaluation component that assesses pose estimator performance:

- **Purpose**: Tests the estimator against ground truth data
- **Key Features**:
  - Loads dataset and ground truth annotations
  - Sends test samples to the estimator
  - Compares predictions with ground truth
  - Calculates accuracy metrics
  - Generates evaluation reports
- **Default Image**: `ghcr.io/opencv/bpc/bpc_tester:latest`
- **Data Access**: Requires mounting of the dataset directory

### ZENOH_CONTAINER ("bpc_zenoh")

The communication middleware that connects the other components:

- **Purpose**: Provides an efficient communication layer
- **Key Features**:
  - Pub/sub messaging system
  - Low-latency data transfer
  - Decouples components for modularity
- **Image**: `eclipse/zenoh:1.2.1`
- **Configuration**: Uses host networking for inter-container communication

### System Integration

The containers work together as an integrated evaluation pipeline:

1. All three containers run in parallel, managed by separate threads
2. They communicate via the host network (using `--network host`)
3. The workflow follows this pattern:
   - Zenoh establishes communication infrastructure
   - Tester loads test cases from the dataset
   - Estimator initializes models and waits for requests
   - Tester sends data to the estimator via Zenoh
   - Estimator processes data and returns pose estimates
   - Tester evaluates results against ground truth

### Resource Requirements

These containers can be resource-intensive:

- **Estimator**: Requires substantial memory and GPU resources for deep learning
- **Tester**: Needs access to dataset files and sufficient RAM for evaluation
- **Zenoh**: Relatively lightweight but critical for system communication

Resource constraints can be added to limit container memory usage and prevent out-of-memory errors.

## OOM - Error 137 out of memory 

### Problem - No soltion yet. 

```
[INFO] [1742398477.822295910] [bpc_pose_estimator]: Pose estimates can be queried over srv /get_pose_estimates.
Non-interactive Docker run failed
 Command '['docker', 'run', '--rm', '-e', 'BOP_PATH=/opt/ros/underlay/install/datasets/', '-e', 'DATASET_NAME=ipd', '--name', 'bpc_tester', '--network', 'host', '-v', '/home/ram/bpc_ws:/opt/ros/underlay/install/datasets', '-v', '/home/ram/bpc_ws:/submission', '23f899a57e08']' returned non-zero exit status 1.
bpc_tester finished with exit code 1 -- stopping others.
Non-interactive Docker run failed
 Command '['docker', 'run', '--rm', '--name', 'bpc_zenoh', '--network', 'host', '86a36af59cff']' returned non-zero exit status 137.
bpc_zenoh finished with exit code 137 -- stopping others.
Non-interactive Docker run failed
 Command '['docker', 'run', '--rm', '--name', 'bpc_estimator', '--network', 'host', '--gpus', 'all', 'bdac47bec806']' returned non-zero exit status 137.
Estimator finished with exit code 137
```

This is caused when the system doesnt meet the minimum requirements defined in the docker image. to overcome this, we need to reduce the resource requirement by passing resource usage limits as arguments. bpc repo page instructs us to install the python client ibpc using pip install bpc which basically installs ibpc.py (https://pypi.org/project/ibpc/, https://github.com/Spinus1/bpc) for simple running of commands like bpc test, bpc estimate etc. 

As a work around to overcome the OOM error, we slightly modify the ibpc.py script to pass memory limits as arguments and use that like 

```
python ibpc.py test ghcr.io/opencv/bpc/bpc_pose_estimator:example ipd
```
instead of 

```
bpc test ghcr.io/opencv/bpc/bpc_pose_estimator:example ipd
```

The modified ibpc.py is committed here in the repo under the folder ibpc. But it still doesn't work because the docker image args are overwritten in the docker file to which we do not have access. So no solution for running this in systems having GPU memory < 8GB. That's a shame :(




