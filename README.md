# Grocery Item Counter using Computer Vision

This repository contains a working prototype for an application that counts items in a grocery shop or inventory warehouse using a camera. The system leverages deep learning (using a pre-trained YOLOv5 model) to detect objects, counts them, and associates each detection with product details from a simulated product database.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Prototype](#running-the-prototype)
- [Deployment](#deployment)
  - [Containerization with Docker](#containerization-with-docker)
  - [Cloud Deployment](#cloud-deployment)
  - [Production Considerations](#production-considerations)
- [Data Engineering & Deep Learning](#data-engineering--deep-learning)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project demonstrates an end-to-end pipeline for counting items using a camera. The application:
- Captures images via a webcam.
- Processes images with a YOLOv5 model (loaded via PyTorch Hub) to detect and count objects.
- Looks up associated product details (brand and product name) using a simulated product database.
- Displays the detection results with bounding boxes and labels.

The system is designed to function even if detailed product information is not available—by simply aggregating the count of similar items.

## Features

- **Real-time Image Capture:** Uses OpenCV to capture images from a webcam.
- **Object Detection:** Utilizes YOLOv5 to identify items in the captured image.
- **Product Counting:** Counts detected items and aggregates results.
- **Product Lookup:** Matches detected labels with a product details dictionary.
- **Visualization:** Annotates and displays images with bounding boxes and labels.

## Project Structure

- `main.py`: Main script to capture an image, perform object detection, count items, and display annotated results.
- `README.md`: This file.
- `requirements.txt`: (Optional) File listing project dependencies.

## Prerequisites

- Python 3.7 or later
- A functional webcam connected to your system

### Python Dependencies

The project relies on the following libraries:
- OpenCV (`opencv-python`)
- PyTorch (`torch` and `torchvision`)
- NumPy (`numpy`)
