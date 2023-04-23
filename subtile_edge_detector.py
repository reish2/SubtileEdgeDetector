import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class SubtileEdgeDetectorConfig:
    """
    SubtileEdgeDetectorConfig is a class that holds configuration parameters for the SubtileEdgeDetector class.
    The parameters control various aspects of the edge detection process, including Gabor filter settings,
    non-maximum suppression window size, and edge length threshold.

    Attributes:
        gabor_kernel_size (int): The size of the Gabor filter kernel. Default is 11.
        gabor_num_filters (int): The number of filters with different orientations. Default is 16.
        non_max_suppression_winsize (int): The window size for non-maximum suppression. Default is 5.
        edge_length_threshold (int): The minimum length of edges to be considered valid. Default is 10.
    """

    def __init__(self) -> None:
        """
        Initialize the SubtileEdgeDetectorConfig instance.

        This class contains various configuration options for the edge detection algorithm.
        """
        self.gabor_kernel_size = 11
        self.gabor_num_filters = 16
        self.non_max_suppression_winsize = 5
        self.edge_length_threshold = 10


class SubtileEdgeDetector():
    """
    SubtileEdgeDetector is a class for detecting edges in images with subpixel accuracy.
    It uses Gabor filters to detect oriented edges and performs non-maximum suppression
    and subpixel interpolation to refine edge locations. Additionally, it filters edges
    based on their length, discarding those that are shorter than a specified threshold.

    Attributes:
        config (SubtileEdgeDetectorConfig): Configuration parameters for the edge detection algorithm.
    """
    def __init__(self, config: Optional[SubtileEdgeDetectorConfig] = None) -> None:
        """
        Initialize the SubtileEdgeDetector instance.

        This method initializes the configuration and applies any changes.

        :param config: An optional SubtileEdgeDetectorConfig instance to configure the edge detector. If not provided,
                       the default configuration will be used.
        """
        # Initialize configuration
        if config is None:
            self.config = SubtileEdgeDetectorConfig()
        else:
            self.config = config

        # Update configuration by applying changes
        self.apply_config_changes()

    def apply_config_changes(self) -> None:
        """
        Apply configuration changes by creating Gabor filters based on the current configuration settings.

        This method should be called whenever the configuration settings are updated.
        """
        # Create Gabor filters based on the configuration
        self.create_gabor_filters()


    def compute(self, image: np.ndarray):
        """
        Perform edge detection on the input image.

        :param image: The input image as a numpy array.
        :return: A tuple containing:
                 1. Edge magnitudes at subpixel locations (numpy array)
                 2. Binary edge mask (numpy array)
                 3. List of edge contours (list of numpy arrays)
                 4. Edge orientations (numpy array)
        """
        # Convert the image to grayscale if necessary
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image

        # Create meshgrid indices for the image
        self._y0_ind, self._x0_ind = np.meshgrid(np.arange(gray.shape[0]), np.arange(gray.shape[1]), indexing='ij')

        # Compute edge magnitudes and angles
        self.compute_edgel_images(gray)

        # Compute subpixel coordinates for all edges
        self.compute_subpixel_edgels(gray)

        # Compute non-maximum suppression mask
        self.compute_non_max_suppression_mask()

        # Filter valid edges by length threshold
        self.fillet_edges(gray)



    def fillet_edges(self, img_in: np.ndarray) -> None:
        """
        Filter and fillet the edges found in the image.

        This method processes the edgel masks to obtain filtered contours and subpixel coordinates.
        It also draws the filtered contours on a binary mask and displays the results if debug mode is enabled.

        :param img_in: Input image as a numpy array
        """
        # Find contours of the edges
        self._valid_edgel_mask = self._non_max_suppression_mask & self._edgel_parabola_valid_mask
        edges = self._valid_edgel_mask.astype(np.uint8)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Create a mask with the same dimensions as the image
        self.edgel_mask = np.zeros_like(edges)

        # Initialize the list to store filtered subpixel contour coordinates
        self.edgel_contours_subpix = []

        # Iterate through the contours
        for contour in contours:
            # Check if the contour length is greater than the threshold
            if cv2.arcLength(contour, True) > self.config.edge_length_threshold:
                # Draw valid contours on the binary mask
                cv2.drawContours(self.edgel_mask, [contour], -1, 255, thickness=cv2.FILLED)

                # Extract valid subpixel coordinates and append them to the filtered contour output list
                x = self._edgel_xy_subpix[contour[:, 0, 1], contour[:, 0, 0], 0]
                y = self._edgel_xy_subpix[contour[:, 0, 1], contour[:, 0, 0], 1]
                v = self._valid_edgel_mask[contour[:, 0, 1], contour[:, 0, 0]]
                res = np.zeros((len(v[v]), 1, 2))
                res[:, 0, 0] = x[v]
                res[:, 0, 1] = y[v]
                self.edgel_contours_subpix.append(res)

        #self.edgel_contours_subpix = self.join_contours(self.edgel_contours_subpix, 1)

    def plot_results(self, image: np.ndarray) -> None:
        """
        Plot the results of the SubtileEdgeDetector on a 2x2 grid using matplotlib.

        :param image: The input image as a numpy array.
        """
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))

        # Display edgel_magnitude_subpix
        ax[0, 0].imshow(self.edgel_magnitude_subpix)
        ax[0, 0].set_title("Edgel Magnitude Subpixel")
        ax[0, 0].axis('off')

        # Display edgel_mask
        ax[0, 1].imshow(self.edgel_mask)
        ax[0, 1].set_title("Edgel Mask")
        ax[0, 1].axis('off')

        # Display edgel_contours_subpix
        ax[1, 0].imshow(image)
        for contour in self.edgel_contours_subpix:
            ax[1, 0].plot(contour[:, 0, 0], contour[:, 0, 1], linewidth=0.5)
        ax[1, 0].set_title("Edgel Contours Subpixel")
        ax[1, 0].axis('off')

        # Display edgel_theta
        ax[1, 1].imshow(self.edgel_theta, cmap='jet')
        ax[1, 1].set_title("Edgel Theta")
        ax[1, 1].axis('off')

        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()

    def compute_non_max_suppression_mask(self) -> None:
        """
        Compute the non-maximum suppression mask for the edgel magnitudes.

        This method finds the local maxima of the edgel magnitudes self.edgel_magnitude_subpix along the orthogonal
        direction and sets the corresponding values in the non-maximum suppression mask self._non_max_suppression_mask
        """
        # Set the size of the non-maximum suppression window
        window_size = self.config.non_max_suppression_winsize

        # Initialize the non-maximum suppression mask as a boolean array with the same shape as edgel_magnitude_subpix
        nms_valid = np.ones_like(self.edgel_magnitude_subpix, dtype=bool)

        # Iterate through the window
        for h in np.linspace(-window_size / 2, window_size / 2, window_size, endpoint=True):
            # Calculate x and y indices by adding h * orthogonal basis to the initial x and y indices
            x_ind = (self._x0_ind + h * self._edgel_x_ortho_basis).astype(np.float32)
            y_ind = (self._y0_ind + h * self._edgel_y_ortho_basis).astype(np.float32)

            # Sample the magnitude values at the calculated x and y indices
            mag_sample = cv2.remap(self._edgel_magnitude, x_ind, y_ind, cv2.INTER_LANCZOS4)

            # Update the non-maximum suppression mask by retaining only the local maxima
            nms_valid = nms_valid & (self.edgel_magnitude_subpix >= mag_sample)

        # Update the non-maximum suppression mask
        self._non_max_suppression_mask = nms_valid

    def compute_edgel_images(self, gray: np.ndarray) -> None:
        """
        Compute edgel images using the Gabor filter.

        This method computes the edge orientation and magnitude using the Gabor gradient kernels.
        It also updates the edgel orthogonal basis vector that will be used for image sampling.

        :param gray: Grayscale input image as a numpy array
        """
        # Initialize the tensor to store the results of Gabor filtering
        gabor_img_grad_tensor = np.zeros((gray.shape[0], gray.shape[1], len(self._gabor_grad_kernels)), dtype=gray.dtype)

        # Apply Gabor filtering to the grayscale image using gradient kernels
        for k, kernel in enumerate(self._gabor_grad_kernels):
            depth = -1
            gabor_img_grad_tensor[:, :, k] = np.absolute(cv2.filter2D(gray, depth, kernel))

        # Determine the indices of the maximum values along the third axis (gradient angles axis) of the tensor
        argmax = np.argmax(gabor_img_grad_tensor, axis=2)
        argmax_p1 = argmax + 1
        argmax_n1 = argmax - 1

        # Handle index overflows
        argmax_p1[argmax_p1 >= self.config.gabor_num_filters] = 0
        argmax_n1[argmax_n1 < 0] = self.config.gabor_num_filters - 1

        # Interpolate the "exact" edge orientation and magnitude using a parabolic fit
        y_ind, x_ind = self._y0_ind, self._x0_ind
        argmax_offset, self._edgel_magnitude, mask = self.find_parabola_maximum(gabor_img_grad_tensor[y_ind, x_ind, argmax_n1],
                                                                               gabor_img_grad_tensor[y_ind, x_ind, argmax],
                                                                               gabor_img_grad_tensor[y_ind, x_ind, argmax_p1])

        self.edgel_theta = self._gabor_angles[argmax] + (self._gabor_angles[1] - self._gabor_angles[0]) * argmax_offset * mask.astype(np.float32)

        # Update the edgel orthogonal basis vector for later use
        self._edgel_x_ortho_basis = np.cos(self.edgel_theta)
        self._edgel_y_ortho_basis = np.sin(self.edgel_theta)

    def create_gabor_filters(self) -> None:
        """
        Create a set of Gabor filters with different orientations.

        This method creates a list of Gabor filter kernels and a numpy array of the corresponding angles.
        """
        ksize = self.config.gabor_kernel_size
        num_filters = self.config.gabor_num_filters
        filters = []
        kshape = (ksize, ksize)
        decay = 2.0
        sigma = ksize / (decay * 3.0)
        lbd = ksize / decay
        gamma = 1.0 / decay
        psi = np.pi / 2.0
        ktype = cv2.CV_32F
        angles = np.arange(0, np.pi, np.pi / num_filters)

        for theta in angles:  # Theta is the orientation for edge detection
            kernel = cv2.getGaborKernel(kshape, sigma, theta, lbd, gamma, psi, ktype)
            kernel /= 1.0 * np.absolute(kernel).sum()  # Brightness normalization
            filters.append(kernel)

        self._gabor_grad_kernels, self._gabor_angles = filters, angles


    def compute_subpixel_edgels(self, gray: np.ndarray) -> None:
        """
        Compute subpixel coordinates for all edges in the grayscale image.

        :param gray: The input grayscale image as a numpy array.
        """
        # Create three new meshgrid arrays to sample the magnitude image along the edge orthogonal direction
        x_ind_n1 = (self._x0_ind - self._edgel_x_ortho_basis).astype(np.float32)
        y_ind_n1 = (self._y0_ind - self._edgel_y_ortho_basis).astype(np.float32)
        mag_n1 = cv2.remap(self._edgel_magnitude, x_ind_n1, y_ind_n1, cv2.INTER_LANCZOS4)
        x_ind_p1 = (self._x0_ind + self._edgel_x_ortho_basis).astype(np.float32)
        y_ind_p1 = (self._y0_ind + self._edgel_y_ortho_basis).astype(np.float32)
        mag_p1 = cv2.remap(self._edgel_magnitude, x_ind_p1, y_ind_p1, cv2.INTER_LANCZOS4)

        # Calculate orthogonal maxima offset, subpixel edge magnitude, and parabolic validity mask
        orthogonal_maxima_offset, edge_magnitude_subpix, parabola_valid_mask = self.find_parabola_maximum(mag_n1,
                                                                                                     self._edgel_magnitude,
                                                                                                     mag_p1)
        edge_magnitude_subpix[~parabola_valid_mask] = self._edgel_magnitude[~parabola_valid_mask]

        # Calculate subpixel accurate coordinates for all edges (valid and invalid)
        # Note: All pixels are treated as edges initially, and only some are valid.
        # This approach simplifies the code.
        edge_x_subpix = (self._x0_ind + orthogonal_maxima_offset * self._edgel_x_ortho_basis).astype(np.float32)
        edge_y_subpix = (self._y0_ind + orthogonal_maxima_offset * self._edgel_y_ortho_basis).astype(np.float32)
        edge_xy_subpix = np.dstack((edge_x_subpix, edge_y_subpix))

        # Set results
        self.edgel_magnitude_subpix = edge_magnitude_subpix
        self._edgel_xy_subpix = edge_xy_subpix
        self._edgel_parabola_valid_mask = parabola_valid_mask



    def find_parabola_maximum(self, y_minus1: np.ndarray, y_0: np.ndarray, y_1: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find the maximum value of a parabola given three points (y_minus1, y_0, y_1).

        :param y_minus1: NumPy array representing the y-values of the points to the left of the maximum.
        :param y_0: NumPy array representing the y-values of the points at the maximum.
        :param y_1: NumPy array representing the y-values of the points to the right of the maximum.
        :return: A tuple containing three NumPy arrays representing the x-coordinate (h) and y-coordinate (k) of the maximum,
                 and a boolean array indicating whether the parabola opens downwards and the maximum is within the given points.
        """
        # Ensure the input arrays are NumPy arrays
        y_minus1, y_0, y_1 = np.array(y_minus1), np.array(y_0), np.array(y_1)

        # Check if the input arrays have the same shape
        if y_minus1.shape != y_0.shape or y_minus1.shape != y_1.shape:
            raise ValueError("The input arrays must have the same shape.")

        # Precomputed inverse of matrix A
        A = np.array([[1, -1, 1], [0, 0, 1], [1, 1, 1]])
        A_inv = np.linalg.inv(A)

        # Multiply the inverse of A with y to obtain the coefficients a, b, and c (broadcasting is used)
        a, b, c = np.einsum('ij,klj->ikl', A_inv, np.dstack((y_minus1, y_0, y_1)))

        # Find the vertex (h, k) of the parabola for each element
        h = -b / (2 * a)
        k = a * h**2 + b * h + c

        # Return the maximum value (k) of the parabola for each element, along with x-coordinate (h)
        # and a boolean array indicating whether the parabola opens downwards and the maximum is within the given points
        return h, k, (a < 0) & (np.abs(h) < 0.5)

