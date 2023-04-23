SubtileEdgeDetector
===================

Introduction
------------
SubtileEdgeDetector is a Python library for detecting and extracting subpixel-accurate edges from images. The detectors biggest strength is finding faint and subtile edges that may have one single brightness step change. The library uses Gabor filters and parabolic fitting to achieve high-precision edge detection and provides subpixel coordinates for all detected edges. It is designed to be used in various computer vision tasks, such as feature extraction and camera calibration.

Features
--------
- Faint subpixel-accurate edge detection using Gabor filters and parabolic fitting
- Configurable Gabor filter parameters, including kernel size and number of orientations
- Configurable non-maximum suppression window size
- Configurable edge length threshold
- Debug mode for visualizing intermediate results

Installation
------------
To install the SubtileEdgeDetector library, simply clone the repository and install the required dependencies:

```bash
git clone https://github.com/reish2/SubtileEdgeDetector.git
cd SubtileEdgeDetector
pip install -r requirements.txt
```

Usage
-----
To use the SubtileEdgeDetector library, simply import the SubtileEdgeDetector class and create an instance:

```python
import cv2
from subtile_edge_detector import SubtileEdgeDetector

# Create an instance of the SubtileEdgeDetector class
detector = SubtileEdgeDetector()

# Load an input image (in grayscale or color format)
image = cv2.imread("path/to/your/image.png")

# Perform edge detection
edgel_magnitude_subpix, edgel_mask, edgel_contours_subpix, edgel_theta = detector.compute(image)
```

Configuration
-------------

You can customize the behavior of the SubtileEdgeDetector by modifying its configuration parameters. To do so, create an instance of the SubtileEdgeDetectorConfig class, modify the parameters as needed, and pass the configuration object to the SubtileEdgeDetector class:

```python
from subtile_edge_detector import SubtileEdgeDetector, SubtileEdgeDetectorConfig

# Create a custom configuration object
config = SubtileEdgeDetectorConfig()
config.gabor_kernel_size = 15
config.non_max_suppression_winsize = 7
config.edge_length_threshold = 20
config.debug = False

# Create an instance of the SubtileEdgeDetector class with the custom configuration
detector = SubtileEdgeDetector(config)

# Changing the configuration of the detector instance is simple too
detector.config.edge_length_threshold = 10
detector.apply_config_changes() # make sure to call the apply_config_changes() method
```

Contributing
------------

We welcome contributions to the SubtileEdgeDetector project. If you would like to contribute, please fork the repository and submit a pull request with your changes.

License
-------

This project is licensed under the MIT License. See the LICENSE file for details.