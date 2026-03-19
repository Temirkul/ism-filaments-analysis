# Repository Guide

This guide is an extended explanation of the repository for someone who is completely new to the project. It does not replace the original `README.md`; it expands on it.

## 1-Minute Summary

This repository is about estimating filament orientations in interstellar medium images with a deep learning model.

At a high level, the workflow is:

1. Build synthetic training images that contain simple filament-like shapes.
2. Train a U-Net to predict an orientation label for every pixel.
3. Preprocess real astronomical images so they look more like the synthetic training data.
4. Run the model and visualize the predicted orientation map.

The important practical point is that this repository is currently notebook-driven. The reusable Python modules define the data generator, dataset loader, and model, but the end-to-end example workflow lives in notebooks rather than a single command-line training or inference script.

## What The Repository Contains

The current checkout contains:

- source code under `src/`
- a synthetic rectangles dataset under `data/Rects/`
- a train/validation split of that synthetic dataset
- a bundled real-data sample under `data/PlanckData/`
- a pretrained checkpoint under `model_saves/`
- an example notebook under `src/example.ipynb`

From inspection of the current repository contents:

- `data/PlanckData/images/` contains 135 files
- `data/PlanckData/masks/` contains 135 files
- `data/PlanckData/thetas/` contains 135 files
- `data/Rects/256_256_1_10_0_0_0_3.5rects_10000_output/train/imgs/` contains 9000 files
- `data/Rects/256_256_1_10_0_0_0_3.5rects_10000_output/val/imgs/` contains 1000 files

## How To Think About The Project

The repository has four main layers.

### 1. Synthetic data generation

The project does not start from a large labeled real-world orientation dataset. Instead, it creates synthetic images where the orientation is known by construction.

Those synthetic images are made of rotated rectangles that stand in for filaments. Because the rectangles are generated programmatically, the code also knows the target orientation label for each rectangle pixel.

This is handled by `src/data_generator.py`.

### 2. Dataset loading

Once image and label files exist on disk, they are loaded by a PyTorch `Dataset` class.

This is handled by `src/dataset.py`.

### 3. Model definition

The model itself is a U-Net that performs dense per-pixel prediction.

This is handled by `src/model.py`.

### 4. Notebook workflows

The repository currently uses notebooks for actual usage examples:

- `src/example.ipynb` for the included pretrained model and bundled data

## Repository Layout

```text
.
|-- data/
|   |-- PlanckData/
|   |   |-- images/
|   |   |-- masks/
|   |   `-- thetas/
|   `-- Rects/
|       |-- 256_256_1_10_0_0_0_3.5rects_10000/
|       |   |-- imgs/
|       |   `-- masks/
|       `-- 256_256_1_10_0_0_0_3.5rects_10000_output/
|           |-- train/
|           `-- val/
|-- model_saves/
|   `-- unet_256_256_1_10_0_0_0_3.5rects_10000_trial12_33epoch_0.00023451543556706085.pt
|-- src/
|   |-- data_generator.py
|   |-- dataset.py
|   |-- example.ipynb
|   `-- model.py
|-- README.md
|-- REPOSITORY_GUIDE.md
|-- requirements.txt
`-- requirements_ubuntu.txt
```

## The Core Source Files

### `src/data_generator.py`

This file is responsible for creating synthetic data.

The main function is:

```python
generate_dataset(
    num_images=10000,
    min_filaments=1,
    max_filaments=10,
    add_perlin_noise=False,
    vmin=0,
    vmax=0,
    rect_noise_num=0,
    rect_noise_scale_factor=3.5,
    img_width=256,
    img_height=256,
)
```

What it does:

- creates a folder under `data/Rects/`
- generates grayscale images of rotated rectangles
- generates matching mask images that encode orientation information
- optionally adds Perlin noise
- optionally adds additional low-contrast rectangle clutter
- returns the output directory path

Important implementation detail:

- The script comment says it should be run from the repository root.
- When the file is run directly as `__main__`, it also splits the generated dataset with `splitfolders.ratio(...)`.

### `src/dataset.py`

This file defines `ImagesAndMasksDataset`, which is the repository's main dataset loader.

What it does:

- reads grayscale images using OpenCV
- reads grayscale masks using OpenCV
- resizes images and masks if needed
- optionally thresholds masks to binary values
- normalizes image intensities to `[0, 1]` by default
- returns PyTorch tensors

Returned tensor shapes:

- image: `[1, H, W]`
- mask: `[H, W]`

Important non-obvious detail:

- Files are matched by sorting filenames in `image_dir` and `mask_dir`.
- There is no explicit stem matching or filename validation.
- If you add your own data, the sorted order of the image and label folders must stay aligned.

### `src/model.py`

This file defines the neural network.

The architecture is a U-Net made from:

- repeated double convolution blocks
- max pooling in the encoder
- transposed convolutions in the decoder
- skip connections between encoder and decoder stages

The model is configurable through:

- `num_classes`
- `apply_batch_norm`

In the example notebook, the model is created as:

```python
UNet(num_classes=181, apply_batch_norm=False)
```

That means the network outputs 181 channels per pixel, one for each class index from 0 to 180.

Also note:

- `src/model.py` includes a small `main()` function that prints model summaries with `torchinfo.summary`.
- That file is mainly for architecture definition and inspection, not for training.

### `src/example.ipynb`

This is the main user-facing example in the repository.

It does four important things:

1. Loads the pretrained checkpoint.
2. Builds dataloaders for synthetic and Planck data.
3. Runs inference on the validation split.
4. Demonstrates preprocessing and inference on bundled real data.

The notebook includes:

- `from model import UNet`
- `from dataset import ImagesAndMasksDataset`
- dataloader creation for synthetic train and validation data
- dataloader creation for Planck data
- a helper function:

```python
def get_unet_masks(unet: UNet, images: torch.Tensor) -> torch.Tensor:
    unet_raw_masks = unet(images)
    unet_masks = torch.argmax(unet_raw_masks, dim=1)
    return unet_masks
```

This helper is the core inference step. It turns raw network logits into a final per-pixel class image by taking `argmax` along the class dimension.

## What The Data Means

### Synthetic rectangles data

The synthetic data is not just ordinary segmentation data.

The generated mask images represent orientation labels rather than plain foreground/background masks. In practice:

- the input image is grayscale
- the label image stores orientation information as grayscale class IDs
- the U-Net is trained as a per-pixel classifier

Because the model is configured with `181` output classes, the effective label space is treated as `0..180` in the notebook visualizations.

If you need the exact synthetic label mapping logic, the relevant place to inspect is `get_color_mask(...)` in `src/data_generator.py`.

### Real Planck data

The bundled real dataset has three parallel folders:

- `images/`: real grayscale inputs
- `masks/`: binary masks derived from those images
- `thetas/`: angle label images

The example notebook uses:

```python
planck_images_dataset = ImagesAndMasksDataset(planck_image_dir, planck_theta_dir)
planck_masks_dataset = ImagesAndMasksDataset(planck_mask_dir, planck_theta_dir)
```

That means the second tensor in those Planck dataloaders is the theta label image, not a foreground/background mask.

This is an important detail for anyone reading the notebook quickly, because variable names like `masks_planck` can make it look like the second tensor is a binary mask when it is actually the orientation label image.

## How To Use The Repository Today

## Option 1: Use the included pretrained model

This is the simplest way to get started.

### Step 1. Install dependencies

Use Python 3.11.x and install one of:

- `pip install -r requirements.txt` on Windows
- `pip install -r requirements_ubuntu.txt` on Ubuntu

### Step 2. Confirm the checkpoint path

The example notebook expects:

```python
path = "../model_saves/unet_256_256_1_10_0_0_0_3.5rects_10000_trial12_33epoch_0.00023451543556706085.pt"
```

That file is currently present in `model_saves/` in this repository snapshot. If it is missing in another checkout, use the download link from the original README.

### Step 3. Open `src/example.ipynb`

The notebook assumes:

- imports like `from model import UNet`
- relative paths such as `../data/...`

So if your notebook environment fails on imports or paths, make sure the working directory is effectively `src/`.

### Step 4. Run the initialization cells

The notebook:

- loads the model
- loads checkpoint weights
- creates synthetic dataloaders
- creates Planck dataloaders

### Step 5. Run validation inference

This lets you see how predictions look on the synthetic validation split, which is the closest thing to a clean sanity check included in the repository.

### Step 6. Run real-data preprocessing and inference

The notebook shows that raw real inputs are preprocessed before being sent into the network.

For the bundled Planck example, the preprocessing includes:

- morphological top-hat with a `7x7` elliptical kernel
- renormalization
- morphological open and dilation with a `3x3` elliptical kernel

The processed images are then passed to the U-Net and displayed with a color scale spanning `0..180`.

## Option 2: Generate synthetic data yourself

If you want to create more synthetic samples, run:

```bash
python src/data_generator.py
```

Run it from the repository root.

That will:

- generate the default 10,000-sample dataset
- save images and masks under `data/Rects/...`
- create a 90/10 train/validation split

If you want different settings, call `generate_dataset(...)` directly with different arguments.

## Option 3: Apply the model to your own astronomical data

The current repository does not provide a polished prediction script, but the intended process is clear from the notebooks:

1. Load your image into a 2D array.
2. Normalize and preprocess it.
3. Convert it into a float tensor.
4. Add batch and channel dimensions so the shape becomes `[1, 1, H, W]`.
5. Run the U-Net.
6. Take `argmax` over the class dimension.
7. Visualize or save the predicted angle map.

## What Is Missing Or Still Rough

A new contributor should know these limitations up front.

### There is no packaged training script

The repository has:

- synthetic data generation
- dataset loading
- model definition
- inference examples

But it does not currently contain a clean `train.py` or `predict.py`.

### Utility code still lives in notebooks

Some helper logic still lives in notebooks rather than reusable library modules.

### Real-data preprocessing is essential

The project is not just "load image -> run network".

The model was trained on very simple synthetic images, so real inputs need preprocessing that pushes them closer to that training distribution. That is why the notebooks spend real effort on morphology and thresholding.

### The dataset class is intentionally simple

`ImagesAndMasksDataset` is small and easy to understand, but it also means:

- no explicit filename matching
- no built-in metadata handling
- no special support for FITS data
- no augmentations in the current file

## Suggested Mental Model For A New User

If you want a simple sentence that explains the whole repository, use this:

This repository trains and applies a U-Net that converts filament-like grayscale images into dense orientation-class maps.

Everything else in the codebase supports one of these steps:

- make labeled synthetic data
- load image/label pairs
- define the network
- preprocess real data so the network can use it
- visualize the predicted orientations

## Where To Start If You Want To Extend The Project

If you plan to work on the repository rather than just run the example, the cleanest next improvements would be:

1. Move notebook helper functions into `src/utils.py`.
2. Add a real `train.py`. (Actually, this is not really necessary, since the .pt trained model weights are made available. In any case, I have the training script, if you need it, you can contact me.)
3. Add a real `predict.py`.
4. Add explicit filename consistency checks in `ImagesAndMasksDataset`.
5. Turn the preprocessing functions into reusable modules.

That would make the repository much easier to maintain and much easier for new users to adopt.
