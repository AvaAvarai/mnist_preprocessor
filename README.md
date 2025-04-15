# mnist_preprocessor

MNIST OCR image preproccesor program in Python

A GUI application for visualizing and preprocessing MNIST handwritten digit images with custom convolution kernels.

## Features

- Display 10 sample images from each MNIST class (0-9)
- Refresh button to show new random samples
- Customizable preprocessing matrix with adjustable:
  - Kernel size (3x3 or 5x5)
  - Stride (1, 2, or 3)
  - Individual kernel element values
- Predefined kernel presets (Identity, Edge Detection, Blur, Sharpen)
- Side-by-side visualization of original and processed images

## Requirements

- Python 3.6+
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Pillow
- Tkinter (included with most Python installations)

## Installation

```bash
pip install torch torchvision numpy matplotlib pillow
```

## Usage

Run the program with:

```bash
python main.py
```

The MNIST dataset will be automatically downloaded on first run.

## How to Use

1. The application will initially display 10 random samples for each of the 10 digit classes (0-9)
2. Click "Refresh Samples" to get new random samples
3. Configure the convolution kernel:
   - Select kernel size (3x3 or 5x5)
   - Choose stride (1, 2, or 3)
   - Either input values manually in the kernel matrix or select a preset
4. Click "Apply Preprocessing" to see the processed images
5. Compare the original images (top) with the processed images (bottom) 
