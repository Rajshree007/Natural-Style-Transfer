# Neural Style Transfer (PyTorch)

A PyTorch implementation of **Neural Style Transfer (NST)** — a deep learning technique that combines the *content* of one image with the *artistic style* of another, based on the approach introduced by Gatys et al. This implementation uses a pre-trained **VGG-19** network to extract content and style representations and iteratively optimizes an input image to match both.

## How It Works

1. **Feature Extraction** — A pre-trained VGG-19 network (frozen weights) is used as a fixed feature extractor. Convolutional layers at various depths capture increasingly abstract representations of the image.
2. **Content Representation** — Feature maps from a deep layer (`conv_4`) are used to preserve the structural content of the content image via mean squared error (MSE) loss.
3. **Style Representation** — Feature correlations, captured using **Gram matrices**, are computed at multiple layers (`conv_1` through `conv_5`) to capture texture, color, and pattern information independent of spatial layout.
4. **Optimization** — Starting from a copy of the content image, the pixel values of the *input image itself* (not the network weights) are iteratively updated using **L-BFGS** optimization to minimize a weighted combination of content loss and style loss.
5. **Output** — After a set number of optimization steps, the resulting image blends the content structure of the content image with the visual style of the style image.

## Features

- Uses a pre-trained VGG-19 model as a fixed feature extractor
- Custom `ContentLoss` and `StyleLoss` modules injected directly into the network pipeline
- Gram matrix-based style loss computation
- Automatic GPU/CPU device detection
- Image loading, preprocessing, and visualization utilities built with `torchvision` and `matplotlib`
- Configurable number of optimization steps, style weight, and content weight

## Requirements

- Python 3.x
- PyTorch
- Torchvision
- Pillow (PIL)
- Matplotlib

Install dependencies:

```bash
pip install torch torchvision pillow matplotlib
```

## Project Structure

```
.
├── nst.py            # Main script for running neural style transfer
└── images/
    ├── picasso.jpg    # Style image
    └── dancing.jpg    # Content image
```

> **Note:** The content and style images must be placed in an `images/` directory and must be the **same size**, as required by the network's fixed input dimensions.

## Usage

1. Place your content and style images in the `images/` directory.
2. Update the image filenames in `nst.py` if needed:
   ```python
   style_img = image_loader(image_directory + "picasso.jpg")
   content_img = image_loader(image_directory + "dancing.jpg")
   ```
3. Run the script:
   ```bash
   python nst.py
   ```
4. The script will display the style image, content image, the initial input image, and the final stylized output using `matplotlib`.

## Configuration

Key parameters can be adjusted in the `run_style_transfer` function call:

| Parameter | Description | Default |
|---|---|---|
| `num_steps` | Number of optimization iterations | `3000` |
| `style_weight` | Weight applied to style loss | `1,000,000` |
| `content_weight` | Weight applied to content loss | `1` |

Increasing `num_steps` generally produces sharper, more refined results at the cost of longer runtime.

## Output

The final output is a new image combining the content layout of `content_img` with the style patterns of `style_img`, displayed via `matplotlib` at the end of execution.

## Acknowledgements

This implementation follows the approach described in:
> Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). *A Neural Algorithm of Artistic Style.*

Built using PyTorch's official Neural Style Transfer tutorial as a reference.

## Future Improvements

- [ ] Save output image to disk automatically instead of only displaying it
- [ ] Support configurable image paths via command-line arguments
- [ ] Add option to start from white noise instead of the content image
- [ ] Support batch processing of multiple style/content pairs
