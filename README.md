# Multi-Mask Overlay Viewer

This Python application is a GUI-based tool for visualizing and interacting with multiple mask overlays on medical images, built using **Tkinter** and **SimpleITK**. It allows users to load, display, and manipulate mask data with features like slice navigation and confidence threshold adjustments.

## Features

- **Multiple Mask Overlays**:
  - Load binary masks for different categories (e.g., UNC, Physician A, B, C, D).
  - Combine selected masks into a single overlay for better visualization.
  
- **Slice Navigation**:
  - Scroll through slices of 3D mask data using a vertical scrollbar.

- **Confidence Level Adjustment**:
  - Adjust the confidence level threshold of mask visualization using a horizontal slider.

- **Dynamic Visualization**:
  - Masks are displayed interactively on a resizable canvas using Tkinter and PIL.

## Requirements

- Python 3.x
- Required libraries:
  - `tkinter`
  - `SimpleITK`
  - `numpy`
  - `Pillow`
