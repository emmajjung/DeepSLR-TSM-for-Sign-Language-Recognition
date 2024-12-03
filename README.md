# DeepSLR: TSM for Sign Language Recognition

## Project Overview

DeepSLR is an optimized framework that leverages the Temporal Shift Module (TSM) for efficient and accurate sign language recognition (SLR). This project aims to create a lightweight model suitable for deployment on IoT devices or microcontrollers while maintaining high accuracy in recognizing American Sign Language (ASL) gestures.

## Content

- [Background](#background)
- [Features](#features)
- [Code](#code)
- [Data](#data)
- [Pretrained Models](#pretrained-models)
- [Future Work](#future-work)
- [Contributors](#contributors)

## Background

American Sign Language (ASL) is the third most common language in the United States, with over half a million users. Sign Language Recognition (SLR) is crucial because it can bridge the communication gap between those with hearing impairments and those without. Temporal Shift Module (TSM) offers a promising approach for SLR due to its ability to extract insights from video streams with high accuracy and low computational cost.

## Features

- Efficient image understanding using TSM
- Lightweight model suitable for IoT devices and microcontrollers
- High accuracy in recognizing ASL gestures

## Code

To train and evaluate the model:
```
python main.py
```

## Data

The project uses Roboflow's [American Sign Language Letters Dataset (v1)](https://public.roboflow.com/object-detection/american-sign-language-letters/1.). The dataset is composed of images with their respective annotations. They are separated for training, validation, and testing.

## Pretrained Models

Training models may be computationally expensive. Here we provide some of the pretrained models.

## Future Work 

- Test current model with extensive custom imagery
- Adapt current model to process live feed from camera
- Analyze performance of model to determine ideal hardware
- Continue developing a model to process videos of not just sign language letters
- Pinpoint real-world use cases and tune the model to meet the specific needs of individual users

## Contributors
- Emma Jung
- Crystal Liang
- Ashwini Suriyaprakash
- Jonas Rajagopal