# DCGAN on Fashion-MNIST

**Name:** Sanskar Ramesh Meharkar
**PRN:** 202401110029
**Batch:** A1

Implementation and training of a Deep Convolutional Generative Adversarial Network (DCGAN) on the Fashion-MNIST dataset (60,000 grayscale 28×28 clothing images, 10 classes). The trained generator produces synthetic clothing images, and performance is evaluated using the loss curves, discriminator accuracy and visual quality of the generated images.

## Colab Notebook
[Open in Google Colab](https://colab.research.google.com/drive/17G6V_50h4T7OhZMBPm-9isaQUagYyxqJ)

## Model Summary
- **Discriminator:** 2 strided Conv2D layers (64 filters, 3×3) with LeakyReLU(0.2) and Dropout(0.4), then a Dense sigmoid output (~40.7K parameters)
- **Generator:** Dense layer reshaped to 7×7×128, then 2 Conv2DTranspose layers (128 filters, 4×4, stride 2) with LeakyReLU(0.2), and a Conv2D (7×7) tanh output (~1.16M parameters)
- **Latent dimension:** 100
- **Loss / optimizer:** Binary cross-entropy, Adam (lr = 0.0002, β1 = 0.5)
- **Training:** 50 epochs, batch size 128 (64 real + 64 fake per discriminator update)

## Results
- **Image quality:** The generator produces recognisable Fashion-MNIST items (trousers, sneakers, boots, jumpers, shirts) with good variety and no obvious mode collapse. Some samples are blurry or slightly malformed.
- **Loss curves:** Discriminator and generator losses plateau near 0.69–0.70 after the first few epochs, so training is stable and does not diverge.
- **Discriminator accuracy:** Accuracy on real images is about 20–40% and on fake images about 70–95%, so the discriminator is not perfectly balanced.
- **Latent space:** Interpolating between two latent vectors gives smooth transitions between images.

## Limitations
- No quantitative image metric (such as FID) was used; evaluation is visual plus loss and accuracy curves.
- The generator has no BatchNorm, and some generated images are imperfect.

## Tech Stack
Python, TensorFlow / Keras, NumPy, Matplotlib, Pandas

## How to Run
1. Open the Colab notebook above.
2. Run all cells in order (Runtime → Run all).
3. Generated image snapshots and loss plots are saved during training.
