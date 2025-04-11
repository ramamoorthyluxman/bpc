# SAM Fine-tuning 

This project provides a complete pipeline for fine-tuning the Segment Anything Model (SAM) from Meta AI on custom datasets with polygon annotations.

## Overview

The Segment Anything Model (SAM) is a powerful foundation model for image segmentation. This project allows you to fine-tune SAM on your custom dataset, making it more effective for your specific segmentation tasks.

The project is organized into multiple components:

1. **Data Preparation**: Convert polygon annotations to binary masks
2. **Dataset Loading**: Create PyTorch datasets for training
3. **Model Definition**: Modify SAM for fine-tuning
4. **Training Pipeline**: Comprehensive training loop with visualization

## Requirements

- Python 3.8+
- PyTorch 1.12+
- segment-anything (Meta AI's SAM implementation)
- numpy
- opencv-python
- matplotlib
- tqdm
- PIL

Install dependencies with:

```bash
pip install torch torchvision numpy opencv-python matplotlib tqdm pillow
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Project Structure

```
/
├── scripts/
│   ├── 1_data_preparation.py    # Process annotations to masks
│   ├── dataset.py               # Dataset class for SAM
│   ├── model.py                 # Fine-tunable SAM model
│   ├── train.py                 # Training loop
│   └── predict.py               # Inference with fine-tuned model
├── data/
│   ├── annotations/             # Input JSON annotations
│   ├── images/                  # Input images
│   └── sam_dataset/             # Processed data
├── results/                     # Training outputs
└── README.md
```

## Dataset Format

Your input dataset should be organized as follows:

```
/path/to/dataset/
├── annotations/                 # JSON annotation files
│   ├── image1.json
│   ├── image2.json
│   └── ...
└── images/                      # Corresponding image files
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

Each JSON annotation should follow this format:

```json
{
  "version": "4.5.6",
  "flags": {},
  "shapes": [
    {
      "label": "object_class",
      "points": [[x1, y1], [x2, y2], ...],
      "group_id": null,
      "shape_type": "polygon",
      "flags": {}
    },
    ...
  ],
  "imagePath": "image_filename.jpg",
  "imageData": null,
  "imageHeight": 500,
  "imageWidth": 800
}
```

## Usage

### 1. Data Preparation

Process your dataset to convert polygon annotations to binary masks:

```bash
python 1_data_preparation.py --data_dir /path/to/dataset --output_dir ./data/sam_dataset --visualize
```

Options:
- `--data_dir`: Path to your dataset containing annotations and images folders
- `--output_dir`: Where to save processed data
- `--visualize`: Enable to visualize masks during conversion
- `--num_samples`: Limit number of samples to process (optional)

### 2. Test Dataset Loading

Verify your dataset is properly loaded:

```bash
python dataset.py --data_dir ./data/sam_dataset --num_samples 3
```

Options:
- `--data_dir`: Path to processed dataset
- `--num_samples`: Number of samples to visualize
- `--point_selection`: Method to select points from masks (center, random, bbox)
- `--num_points`: Number of points per mask

### 3. Fine-tune SAM

Train the model on your dataset:

```bash
python train.py \
  --data_dir ./data/sam_dataset \
  --checkpoint /path/to/sam_vit_h_4b8939.pth \
  --model_type vit_h \
  --output_dir ./results \
  --freeze_image_encoder \
  --batch_size 1 \
  --num_epochs 10 \
  --lr 1e-5
```

Key options:
- `--data_dir`: Path to processed dataset
- `--checkpoint`: Path to SAM checkpoint file
- `--model_type`: SAM model type (vit_h, vit_l, vit_b)
- `--freeze_image_encoder`: Flag to freeze image encoder (recommended)
- `--batch_size`: Batch size for training
- `--num_epochs`: Number of training epochs
- `--lr`: Learning rate

### 4. Inference (Optional)

Use your fine-tuned model for segmentation:

```bash
python predict.py \
  --checkpoint ./results/vit_h_*/best_model.pth \
  --model_type vit_h \
  --image_path /path/to/image.jpg
```

## Training Strategy

The default training strategy:

1. Freeze the image encoder (which contains most parameters)
2. Fine-tune the prompt encoder and mask decoder
3. Use a combination of Dice loss and Focal loss
4. Use center points of masks as prompts for SAM

This approach achieves good results while being computationally efficient.

## Tips

- **Memory Usage**: SAM is memory-intensive. Use batch size of 1 if you encounter memory issues.
- **Image Size**: All images are resized to 1024×1024 for SAM compatibility.
- **Training Time**: Fine-tuning might take several hours depending on dataset size.
- **Points Selection**: Try different point selection strategies to see what works best for your dataset.

## Acknowledgments

This project builds upon Meta AI's Segment Anything Model. The original SAM repository can be found at [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything).

## License

This project is provided under the MIT License.