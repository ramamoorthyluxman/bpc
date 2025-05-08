# 3D Model Rotation Renderer

This tool renders 3D models from different viewpoints by applying random rotations to the model. It generates multiple images from different angles and saves the corresponding rotation matrices. This is particularly useful for creating training datasets for pose estimation tasks in computer vision and robotics.

## Features

- Renders 3D models (PLY format) from multiple viewpoints using random rotations
- Generates high-quality renders using Open3D's offscreen rendering
- Saves all rotation matrices to a text file for later use in pose estimation
- Customizable number of poses to generate
- Proper camera positioning that adapts to model size

## Installation

### Requirements

1. Python 3.6+
2. Open3D
3. NumPy
4. SciPy
5. Pillow (PIL)

Install the required packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

Basic usage:

```bash
python render_rotations.py --model path/to/your/model.ply --output rendered_outputs --poses 100
```

### Arguments

- `--model`: Path to the 3D model file (PLY format) [required]
- `--output`: Output directory for rendered images [default: 'rendered_images']
- `--poses`: Number of poses to generate [default: 100]

## Output

The script creates the following directory structure:

```
output_directory/
├── model_name/
│   ├── rot_000.png
│   ├── rot_001.png
│   ├── ...
│   └── rot_099.png
└── rotation_matrices/
    └── model_name.txt
```

- The `model_name` directory contains the rendered images from different viewpoints
- The `rotation_matrices` directory contains text files with the corresponding rotation matrices
- The first rotation is always the identity matrix (no rotation)

## Example

```bash
python render_rotations.py --model objects/teapot.ply --output dataset --poses 50
```

This will:
1. Load the teapot model
2. Generate 50 images with different rotations
3. Save the images to `dataset/teapot/`
4. Save the rotation matrices to `dataset/rotation_matrices/teapot.txt`

## Applications

This tool is particularly useful for:

- Training pose estimation models
- Creating datasets for 3D object recognition
- Generating synthetic data for computer vision research
- Point cloud registration testing and evaluation

## Notes

- The first rotation is always the identity matrix to include a canonical view
- Rendering quality can be adjusted by modifying the renderer settings in the code
- Lighting parameters can be customized in the script if needed

## License

[MIT License](LICENSE)