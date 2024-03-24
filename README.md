# Simple Neural Network Implementation in C using Intel Intrinsics (AVX2)

Simple (deep feed-forward) neural network implementation in C using Intel Intrinsics (AVX2).

## Program Usage & Compilation

I only tested this in Linux (Fedora 39). You need to have a x86 64-bit CPU with AVX2 to compile main.cpp. 
You can compile it using gcc as below (navigate to **inferece_impl** directory first):
`gcc main.cpp -o fashion_classifier -lm -lstdc++ -march=native -std=c++17 -O3`

Usage:
0: No SIMD
1: SIMD
image_path: See [stbi](https://github.com/nothings/stb) for supported formats. Make sure the image is 28x28.
`./fashion_classifier <image_path> <0|1>`

## Project Structure

- **python** subdirectory contains the code for training the neural network and exporting the model parameters in a binary format.
- **inference_impl** subdirectory contains the C implementation. C implementation for inference has two versions, SIMD & no SIMD. [stbi](https://github.com/nothings/stb) is used for loading images (.jpg, .png, etc.).

## Neural Network

I followed the PyTorch tutorial in official docs. Model takes in a 28x28 pixel grayscale image, containing various fashion items (sneakers, dress, pullover, T-Shirt, etc.) and classifies it accordingly. I only changed the optimizer to Adam from SGD different from the tutorial, which seemed to boost the accuracy significantly.

