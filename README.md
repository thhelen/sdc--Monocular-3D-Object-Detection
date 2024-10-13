# SDC: Monocular 3D Object Detection with Robustness to Adverse Weather Conditions
## Introduction
Monocular 3D object detection is a crucial task for autonomous vehicles, providing essential information for decision-making, route planning, and safety measures. Unlike traditional stereo-based or LiDAR-based methods, monocular detection is cost-effective, computationally efficient, and power-efficient.

MonoCon is a high-performance monocular 3D object detection model that achieves outstanding results without relying on additional sensor data like LiDAR or depth information. The model is particularly efficient for detecting cars and performs exceptionally well in the KITTI benchmark.

This repository contains code for fine-tuning MonoCon to enhance its performance in adverse weather conditions, such as fog. The fine-tuning includes improvements to the learning rate scheduler and data augmentation strategies.

## Key Features
1] Monocular Object Detection: Achieve state-of-the-art 3D object detection from a single RGB image.

2] Cost-Effective & Scalable: Monocular systems are cheaper and less complex compared to stereo or LiDAR systems.

3] Robust to Adverse Weather: Fine-tuned to perform better in adverse weather conditions such as fog, thanks to augmented training data and optimized training techniques.

4] Efficient Training with OneCycleLR: Improved training strategy using the OneCycleLR learning rate scheduler for faster convergence.

5] KITTI Benchmark Leader: MonoCon achieves top performance, particularly in car detection tasks in the KITTI dataset.

## Model Architecture
MonoCon uses a convolutional neural network (CNN) backbone to process RGB images. It integrates auxiliary monocular contexts during the training phase and outputs key parameters for 3D object localization. The training process is end-to-end, and auxiliary branches are discarded after training to ensure fast inference.

## Enhancements in This Repository
1] Learning Rate Optimization: The default cyclic learning rate scheduler has been replaced with the OneCycleLR method, leading to faster convergence and improved model performance.

2] Weather Augmentation: The KITTI dataset is augmented with fog-like effects, simulated using the Monodepth2 model for depth estimation, which helps in training the model to better handle foggy conditions.

3] Data Augmentation with Fog Simulation: Images are augmented with fog effects using the technique described in [8]. This improves the model's robustness when detecting objects in hazy or foggy environments.

## Prerequisites
Python 3.x

PyTorch

Albumentations (for data augmentation)

Monodepth2 (for depth map estimation)

## Results
The results of the fine-tuned MonoCon model on normal and adverse weather conditions are as follows:

![image](https://github.com/user-attachments/assets/d734e9f2-1687-469b-a6e5-93b4fb4ac288)

Normal Weather: Figure 4 shows accurate bounding boxes with strong alignment to ground truth.

Adverse Weather (Fog): Figure 5 demonstrates that the model is resilient to fog, maintaining decent object detection accuracy despite compromised visibility.


