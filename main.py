# Copyright (C) 2023 Reish2
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import cv2
import numpy as np
from subtile_edge_detector import SubtileEdgeDetector, SubtileEdgeDetectorConfig

img1 = cv2.imread("./data/edge_detector_test.png").astype(np.float32)/255.0
img2 = cv2.imread("./data/gates.png").astype(np.float32)/255.0
img3 = cv2.imread("./data/Melbourne-Edge-test-of-contrast-sensitivity.png").astype(np.float32)/255.0


# Create a custom configuration object
config = SubtileEdgeDetectorConfig()
# config.gabor_kernel_size = 15
# config.non_max_suppression_winsize = 7
# config.edge_length_threshold = 20
# config.debug = True

# Create an instance of the SubtileEdgeDetector class with the custom configuration
detector = SubtileEdgeDetector(config)
detector.compute(img1)
detector.plot_results(img1)

detector.compute(img3)
detector.plot_results(img3)

detector.compute(img2)
detector.plot_results(img2)
