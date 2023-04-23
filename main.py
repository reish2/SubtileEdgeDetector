import cv2
import numpy as np
from subtile_edge_detector import SubtileEdgeDetector, SubtileEdgeDetectorConfig

img1 = cv2.imread("./data/Melbourne-Edge-test-of-contrast-sensitivity.png").astype(np.float32)/255.0
img2 = cv2.imread("./data/gates.png").astype(np.float32)/255.0


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

detector.compute(img2)
detector.plot_results(img2)