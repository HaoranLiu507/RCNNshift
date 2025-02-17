import cv2
import numpy as np
import torch
import torch.cuda
import torch.nn.functional as F
import os
import copy
from tqdm import tqdm

device = torch.device("cuda:0")


# device = torch.device("cpu")

class RCNNshift:
    """
    The RCNNshift class is designed to provide trackers for moving object tracking.
    """

    # def __init__(self, weight, batch_size, select_tracker, perform, depth):
    def __init__(self, weight, batch_size, isColor, select_tracker, perform, depth=3):
        """
        Initialize the RCNNshift object with the given parameters

        :param select_rect: Methods of the to be tracked object selection (Optional: "input" or "mouse")
                            If "input", user needs to type in information of the tracking window;
                            if "mouse", user needs to select an area on the first frame of video as the tracking window.
        :param weight: Determine the relative importance between the RCNN feature channel and the original image
                            channel in the feature-enhanced space. The larger the weight, the higher the importance of
                            the RCNN feature channel.
        :param batch_size: Number of images used for RCNN batch processing.
        :param isColor: Whether the input video is in color.
        :param select_tracker: Variable used to select the tracking algorithm (Optional: "RCNNshift", "RCNNshift_3D" or "meanshift")
        :param perform: Whether to display the tracking effect in real-time (Optional: "live" or "local")
                        If "live", the tracking effect is displayed in real-time, and the tracking results is
                        saved in the form of location of tracking window;
                        if "local", after saving the information of the tracking window, a tracking video is saved and
                        the tracking effect is displayed in the end.
        :param depth: The number of video frames used for 3DRCNN temporal feature extraction,
                      for example: depth=3 indicates the use of the current frame along with two adjacent frames for ignition.
        :param track_rect: Tracking window.
        :param track_ROI: ROI of the image selected by the tracking window.
        :param gaussian_kernel_matrix: Two-dimensional gaussian kernel for weight connection.
        :param gaussian_kernel_matrix3D: Two-dimensional gaussian kernel for weight connection.
        :param random_inactivation_probability_matrix: Two-dimensional gaussian kernel for random inactivation probability.
        """
        self.select_rect = None
        self.weight = weight
        self.batch_size = batch_size
        self.isColor = isColor
        self.select_tracker = select_tracker
        self.perform = perform
        self.depth = depth
        self.name = None
        self.video_path = None
        self.track_rect = None
        self.track_ROI = None
        self.gaussian_kernel_matrix = None
        self.gaussian_kernel_matrix3D = None
        self.random_inactivation_probability_matrix = None

    def track(self, video_path, name, select_rect, ROI_region):
        """
        Select the target ROI (Region of Interest) in the first frame of a video and do ROI tracking based on the
        chosen tracking algorithm.
        """
        # To be tracked video
        self.video_path = video_path
        self.name = name
        self.select_rect = select_rect

        # Target ROI selection
        self.loc(ROI_region)

        # Tracking based on the selected algorithm
        if self.select_tracker == 'RCNNshift_3D':

            ignition = self.ignition_3D()
            wei = self.wei()
            hist_first = self.hist_first(wei, ignition)

            # Perform moving object tracking based on the RCNNshift_3D
            self.RCNNshift_track(ignition, hist_first, wei)

            if self.perform == 'local':
                self.show()
            else:
                pass

        elif self.select_tracker == 'meanshift':

            # Perform moving object tracking based on the meanshift
            self.meanshift()

            if self.perform == 'local':
                self.show()
            else:
                pass

        elif self.select_tracker == "RCNNshift":
            ignition = self.ignition()
            wei = self.wei()
            hist_first = self.hist_first(wei, ignition)

            # Perform moving object tracking based on the RCNNshift
            self.RCNNshift_track(ignition, hist_first, wei)

            if self.perform == 'local':
                self.show()
            else:
                pass

    def loc(self, ROIregion):
        """
        This method obtains the size and pixel information of the target ROI in the first frame. It outputs the
        size of the ROI and the pixel information inside the ROI.
        """
        # Read the first frame of the video
        cap = cv2.VideoCapture(self.read_video())
        ret, frame = cap.read()

        # Get the size of the images
        height, width, _ = frame.shape
        print("Image size: {} x {}".format(width, height))

        # Get the total number of frames in the video
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Number of frames: {}".format(num_frames))

        # Release the video capture object
        cap.release()

        global first_frame
        first_frame = frame

        # Manually select the target window using the mouse
        if self.select_rect == "mouse":
            self.track_rect = self.mouse_select(first_frame)

        # Manually input the target window
        if self.select_rect == "input":
            arr = input("input rect:")
            nums = [int(float(n)) for n in arr.split()]
            self.track_rect = nums

        if self.select_rect == "batch_input":
            arr = ROIregion
            nums = [int(float(n)) for n in arr.split()]
            self.track_rect = nums

        self.track_ROI = first_frame[int(self.track_rect[1]):int(self.track_rect[1] + self.track_rect[3]),
                         int(self.track_rect[0]):int(self.track_rect[0] + self.track_rect[2])]

    # Methods used for RCNN-based feature extraction
    def ignition(self):
        """
        Generate RCNN ignition maps of a given video
        """

        # Open the video file
        cap = cv2.VideoCapture(self.read_video())

        # Get the number of frames and the frame dimensions
        nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize an array to store the grayscale frames
        video_gray = np.zeros((frame_height, frame_width, nFrames))

        # Read each frame from the video, convert it to grayscale, and store it in video_gray
        for f in range(nFrames):
            ret, frame = cap.read()
            if ret:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                video_gray[:, :, f] = frame_gray

        # Initialize an array to store the ignition maps
        ignition = np.zeros((frame_height, frame_width, nFrames))

        # # RCNN ignition
        for idx in tqdm(range(0, nFrames, self.batch_size)):

            # If the remaining frames are less than the batch size, process them separately
            if self.batch_size > nFrames - idx > 0:
                ignition[:, :, idx:nFrames] = self.RCNN(video_gray[:, :, idx:nFrames].reshape
                                                        ((frame_height, frame_width, nFrames - idx)), nFrames - idx)

            # Process a batch of frames using the RCNN
            else:
                ignition[:, :, idx:idx + self.batch_size] = self.RCNN(video_gray[:, :, idx:idx + self.batch_size],
                                                                      self.batch_size)

        return ignition

    def RCNN(self, images, batch_size):
        """
        This method performs RCNN-based feature extraction for multiple grayscale images (given in a numpy
        ndarray), where the number of images depends on the given batch size. When the process is down, this funtion
        outputs ignition maps that have the same size as the input numpy ndarray.

        :param images: High-dimensional matrix obtained by merging multiple grayscale images
        :param batch_size: Batch processing size for feature extraction
        :param beta: Weighting factor that controls the relationship between feedback and link inputs
        :param alpha_theta: Dynamic threshold decay coefficient
        :param V_theta: Dynamic threshold weighting coefficient
        :param alpha_U: Internal activity decay coefficient
        :param V_U: Internal activity weighting coefficient
        :param t: Number of iterations for RCNN ignition
        :param sigma_kernel:Variance of 2-D Gaussian distribution for Gaussian kernel matrix
        :param sigma_random_inactivation:Variance of 2-D Gaussian distribution for random inactivation probability matrix
        :param size: Gaussian kernel size (size by size)
        :param rgb_range: RGB range of the image/video (eg, 255 for 8 bit images, and 65536 for 16 bit images)
        """
        # Initialize parameters
        self.beta = 0.1
        self.alpha_theta = torch.tensor(0.53)
        self.V_theta = 16
        self.alpha_U = torch.tensor(0.02)
        self.V_U = 1
        self.t = 170
        self.sigma_kernel = 4
        self.sigma_random_inactivation = 5
        self.size = 9
        self.rgb_range = 255

        # Cook the input images in preparation for latter processing
        images = torch.from_numpy(self.images_norm(images)).to(device)

        # Declare the variables and move them to the device
        [h, w, batch_size] = images.shape
        ignition_map = torch.zeros([h, w, batch_size], dtype=torch.float16).to(device)
        U = ignition_map
        threshold = ignition_map + 1
        neuron_output = ignition_map.to(device)
        self.gaussian_kernel_matrix = self.get_gaussian_kernel(dimension=self.size, sigma=self.sigma_kernel)
        self.gaussian_kernel_matrix[int((self.size - 1) / 2), int((self.size - 1) / 2)] = 0
        self.gaussian_kernel_matrix = torch.unsqueeze(self.gaussian_kernel_matrix, dim=0)
        self.random_inactivation_probability_matrix = self.get_gaussian_kernel(dimension=self.size,
                                                                               sigma=self.sigma_random_inactivation)
        weight_default = self.gaussian_kernel_matrix.unsqueeze(0)

        # Ignition iterations
        for i in range(self.t):
            # Generate the random inactivation matrix
            mask = self.random_inactivation(self.size, 0.1, 'Gaussian', 1)

            # Random inactivation
            weight = torch.where(mask, weight_default, torch.zeros_like(weight_default))

            # Link input
            L = F.conv2d(input=neuron_output.reshape([batch_size, 1, h, w]), weight=weight, bias=None, stride=1,
                         padding=self.size // 2, dilation=1, groups=1).squeeze().reshape([h, w, batch_size])

            # Neural internal activity
            U = torch.exp(-self.alpha_U) * U + images * (1 + self.beta * self.V_U * L)

            # Neuron ignition
            neuron_output = (U > threshold).float().to(torch.float16)

            # Update dynamic threshold
            threshold = torch.mul(torch.exp(-self.alpha_theta), threshold) + self.V_theta * neuron_output

            # Sum ignition results
            ignition_map = ignition_map + neuron_output

        return ignition_map.cpu().numpy()

    def ignition_3D(self):
        """
        Generate 3DRCNN ignition maps of a given video.
        """

        # Open the video file
        cap = cv2.VideoCapture(self.read_video())

        # Get the number of frames and the frame dimensions
        nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize an array to store the final ignition maps
        # The array shape is (frame_height, frame_width, nFrames) with float16 precision
        new_ignition = np.zeros((frame_height, frame_width, nFrames), dtype=np.float16)

        # Check whether the input video is in color or grayscale
        if not self.isColor:
            # For grayscale video:
            # Initialize an array to store grayscale frames.
            # Shape: (1, nFrames, frame_height, frame_width)
            video_img = np.zeros((1, nFrames, frame_height, frame_width), dtype=np.float16)

            # Read each frame, convert to grayscale and store it
            for f in range(nFrames):
                ret, frame = cap.read() # Read the current frame
                # Convert the frame from BGR color to grayscale
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Expand dimensions to maintain a channel axis (needed for consistency)
                frame_gray = np.expand_dims(frame_gray, axis=0)
                video_img[:, f, :, :] = frame_gray
        else:
            # For color video:
            # Initialize an array to store color frames.
            # Shape: (3, nFrames, frame_height, frame_width)
            video_img = np.zeros((3, nFrames, frame_height, frame_width), dtype=np.uint8)
            # Read each frame, transpose to reorder channels and store it
            for f in range(nFrames):
                ret, frame = cap.read() # Read the current frame
                # Transpose frame dimensions from (height, width, channels)
                # to (channels, height, width) for consistency in processing
                new_frame = frame.transpose([2, 0, 1])
                video_img[:, f, :, :] = new_frame


            # Release the video capture object now that all frames are loaded
            cap.release()

        # Initialize arrays for:
        # 1. video_packed: a sliding window of frames for 3D RCNN analysis
        # 2. ignition_3d: the raw 3D ignition maps output from the RCNN model
        video_packed = np.zeros((nFrames, video_img.shape[0], self.depth, frame_height, frame_width), dtype=np.float16)
        ignition_3d = np.zeros((nFrames, video_img.shape[0], self.depth, frame_height, frame_width), dtype=np.float16)

        # Pack video frames into a sliding window structure to capture temporal context
        for idx in range(0, nFrames):
            # For the beginning part of the video, use the first "depth" frames for padding
            if idx < int((self.depth - 1) / 2):
                video_packed[idx, :, :, :, :] = video_img[:, 0:(self.depth), :, :]

            # For the ending part of the video, take the last "depth" frames as padding
            elif nFrames - idx < int((self.depth - 1) / 2 + 1):
                a = video_img[:, (nFrames - self.depth):nFrames + 1, :, :]
                video_packed[idx, :, :, :, :] = video_img[:, (nFrames - self.depth):nFrames + 1, :, :]

            # For the middle part of the video, create a window centered at the current frame
            else:
                video_packed[idx, :, :, :, :] = video_img[:,
                                                (idx - int((self.depth - 1) / 2)):(idx + int((self.depth - 1) / 2) + 1),
                                                :, :]

        # Release the raw video frames from memory as they are no longer needed
        video_img = None

        # Process frames in batches using the 3DRCNN model to generate ignition maps
        for idx in tqdm(range(0, nFrames, self.batch_size)):
            # When the remaining frames are fewer than the defined batch size, process them all
            if self.batch_size > nFrames - idx > 0:
                ignition_3d[idx:nFrames, :, :, :, :] = self.RCNN_3D(video_packed[idx:nFrames, :, :, :, :])
            else:
                # Process a batch of frames using the RCNN_3D model
                ignition_3d[idx:idx + self.batch_size, :, :, :, :] = self.RCNN_3D(
                    video_packed[idx:idx + self.batch_size, :, :, :, :])

        # Post-process the 3D ignition maps to extract a 2D ignition map for each frame
        for idx in range(0, nFrames):
            # For the initial frames, adjust the processing index according to the current frame index
            if idx < int((self.depth - 1) / 2):
                new_ignition[:, :, idx] = self.ignition_MAX(ignition_3d[idx, :, :, :, :], idx)

            # For the ending frames, adjust by computing modulo to fit within the depth window
            elif nFrames - idx < int((self.depth - 1) / 2 + 1):
                new_ignition[:, :, idx] = self.ignition_MAX(ignition_3d[idx, :, :, :, :], idx % self.depth)

            # For the middle frames, use the center slice of the temporal window
            else:
                new_ignition[:, :, idx] = self.ignition_MAX(ignition_3d[idx, :, :, :, :], int((self.depth - 1) / 2))

        # Return the final computed ignition maps
        return new_ignition

    def ignition_MAX(self, ignition, fps):

        # Extract the frame corresponding to the given fps from the 3D ignition tensor
        ignition_3D = ignition[:, fps, :, :]
        [channel, h, w] = ignition_3D.shape

        # If the frame has a single channel, remove the channel dimension
        if channel == 1:
            ignition_max = np.squeeze(ignition_3D, axis=0)

        # If the frame has three channels, take the maximum value across the channel dimension.
        if channel == 3:
            ignition_max = np.max(ignition_3D, axis=0)

        return ignition_max

    # 对灰度图像做前后点火图相互影响生成的点火图
    def RCNN_3D(self, image):
        """
         This method performs 3DRCNN-based feature extraction for image patches (constructed by adjacent frames for
         temporal information extraction). When the process is down, this funtion outputs ignition maps that have the
         same size as the input numpy ndarray.

         :param images: High-dimensional matrix obtained by merging multiple grayscale images
         :param batch_size: Batch processing size for feature extraction
         :param beta: Weighting factor that controls the relationship between feedback and link inputs
         :param alpha_theta: Dynamic threshold decay coefficient
         :param V_theta: Dynamic threshold weighting coefficient
         :param alpha_U: Internal activity decay coefficient
         :param V_U: Internal activity weighting coefficient
         :param t: Number of iterations for RCNN ignition
         :param sigma_kernel:Variance of 3-D Gaussian distribution for Gaussian kernel matrix
         :param sigma_random_inactivation:Variance of 3-D Gaussian distribution for random inactivation probability matrix
         :param size: Gaussian kernel size (size by size)
         :param rgb_range: RGB range of the image/video (eg, 255 for 8 bit images, and 65536 for 16 bit images)
         """

        # Initialize parameters
        self.beta = 0.31
        self.alpha_theta = torch.tensor(0.679)
        self.V_theta = 2.8
        self.alpha_U = torch.tensor(0.11)
        self.V_U = 1
        self.t = 16
        self.sigma_kernel = 4
        self.sigma_random_inactivation = 5
        self.size = 9
        self.rgb_range = 255

        # Cook the input images in preparation for latter processing
        image = torch.from_numpy(self.images_norm(image)).to(device)
        [batchsize, c, depth, h, w] = image.shape

        ignition_map = torch.zeros([batchsize, c, depth, h, w], dtype=torch.float16).to(device)
        U = ignition_map
        threshold = ignition_map + 1
        neuron_output = ignition_map.to(device)

        # Generate a 3D Gaussian kernel for convolution using the specified size and kernel sigma.
        self.gaussian_kernel_matrix3D = self.get_3D_gaussian_kernel(dimension=self.size, sigma=self.sigma_kernel)

        # Set the center element of the kernel to zero to avoid self-feedback.
        self.gaussian_kernel_matrix3D[:, int((self.size - 1) / 2), int((self.size - 1) / 2)] = 0

        # Generate a 3D random inactivation probability matrix using a Gaussian kernel with the specified sigma.
        self.random_inactivation_probability_matrix = self.get_3D_gaussian_kernel(dimension=self.size,
                                                                                  sigma=self.sigma_random_inactivation)

        # Prepare the default weight tensor by repeating the 3D Gaussian kernel across channels.
        weight_default_3D = self.gaussian_kernel_matrix3D.repeat(c, c, 1, 1, 1).to(device)

        # Ignition iterations
        for i in range(self.t):

            # Generate the random inactivation matrix
            mask = self.random_inactivation_3D(self.size, 0.1, 'Gaussian', c).to(device)

            # Apply the random inactivation mask: use the default weight where mask is True; otherwise, set weight to zero.
            weight = torch.where(mask, weight_default_3D, torch.zeros_like(weight_default_3D)).to(device)

            # Compute the link input L by performing a 3D convolution of the neuron output with the masked weight.
            L = F.conv3d(input=neuron_output.reshape([batchsize, c, depth, h, w]),
                         weight=weight,
                         bias=None,
                         stride=1,
                         padding=int(self.size // 2),
                         dilation=1,
                         groups=1)

            # Neural internal activity
            U = torch.exp(-self.alpha_U) * U + image * (1 + self.beta * self.V_U * L)

            # Neuron ignition
            neuron_output = (U > threshold).float().to(torch.float16)

            # Update dynamic threshold
            threshold = torch.mul(torch.exp(-self.alpha_theta), threshold) + self.V_theta * neuron_output

            # Accumulate the neuron output into the overall ignition map.
            ignition_map = ignition_map + neuron_output

        # Return the computed ignition map as a numpy array after moving it to the CPU.
        return ignition_map.cpu().numpy()

    @staticmethod
    def get_gaussian_kernel(dimension, sigma):
        """
        Generate two-dimensional Gaussian kernel.

        :param dimension: Gaussian kernel size
        :param sigma: Variance of 2-D Gaussian distribution
        """
        kernel = torch.from_numpy(cv2.getGaussianKernel(dimension, sigma)).to(device)
        transpose_kernel = torch.from_numpy(cv2.getGaussianKernel(dimension, sigma).T).to(device)
        matrix = torch.multiply(kernel, transpose_kernel).to(torch.float16)

        return matrix

    @staticmethod
    def get_3D_gaussian_kernel(dimension, sigma):
        """
           Generate a three-dimensional Gaussian kernel.

           :param dimension: Size of the Gaussian kernel (must be odd)
           :param sigma: Standard deviation of the Gaussian distribution
           :return: A 3D Gaussian kernel tensor
           """

        if dimension % 2 == 0:
            raise ValueError("Dimension must be an odd number.")

        # Create a 1D Gaussian kernel
        ax = np.arange(-dimension // 2 + 1., dimension // 2 + 1.)
        kernel_1d = np.exp(-0.5 * (ax ** 2) / (sigma ** 2))
        kernel_1d /= kernel_1d.sum()  # Normalize

        # Create a 3D Gaussian kernel by outer product
        kernel_3d = np.outer(kernel_1d, kernel_1d)
        kernel_3d = kernel_3d[:, :, np.newaxis]  # Add new axis for 3D
        kernel_3d = kernel_3d * kernel_1d[np.newaxis, np.newaxis, :]  # Outer product for 3D

        # Convert to PyTorch tensor
        kernel_3d_tensor = torch.from_numpy(kernel_3d).to(torch.float16)

        return kernel_3d_tensor

    def images_norm(self, images):
        """
        Convert the pixel values in the image to the ignition type of float, and normalize it into range 0-1.

        :param images: high-dimensional matrix obtained by merging multiple grayscale images
        """
        return images.astype(np.float16) / self.rgb_range

    def random_inactivation(self, dimension, P, flag, batch_size):
        """
        Generate a random inactivation matrix to modulate the weight contribution of neurons. It is composed of 0 and 1,
        where 1 represents that the connection input between the central nerve and the neuron at that location is
        turned on, while 0 represents that the connection input is turned off, also known as neural connection
        random_inactivation.

        :param dimension: Size of weight matrix
        :param P: Random inactivation probability for uniform distribution
        :param flag: Random inactivation type (Optional: "Gaussian" or "uniform")
                     when assigned to "Gaussian", the random inactivation probability follows two-dimensional Gaussian distribution
                     (i.e. the random inactivation probability is proportional to the distance from the central neuron);
                     when assigned to "uniform", the random inactivation probability follows uniform distribution between 0 and 1
                     (i.e. the random inactivation probability is the same across whole kernel)
        :param batch_size: batch processing size
        """

        if flag == 'Gaussian':
            # Cook the random inactivation probability matrix to meet the batch processing requirements
            random_inactivation_probability = self.random_inactivation_probability_matrix.repeat(
                1, batch_size, 1, 1)

            # Normalize the probability into range 0-1
            random_inactivation_probability = random_inactivation_probability / random_inactivation_probability[
                0, 0, int((dimension - 1) / 2),
                int((dimension - 1) / 2)]

            # Generate random number between 0-1
            random_number = torch.rand(1, batch_size, dimension, dimension).to(device)

            # Random inactivation
            matrix = random_number < random_inactivation_probability

        if flag == 'uniform':
            # Generate random number between 0-1
            random_number = torch.rand(1, batch_size, dimension, dimension).to(device)

            # Constant random inactivation probability
            random_inactivation_probability = torch.ones(1, batch_size, dimension, dimension, device=device) * P

            # Random inactivation
            matrix = random_number < random_inactivation_probability

        return matrix

    def random_inactivation_3D(self, dimension, P, flag, channel):
        """
        Generate a random inactivation matrix to modulate the weight contribution of neurons. It is composed of 0 and 1,
        where 1 represents that the connection input between the central nerve and the neuron at that location is
        turned on, while 0 represents that the connection input is turned off, also known as neural connection
        random_inactivation.

        :param dimension: Size of weight matrix
        :param P: Random inactivation probability for uniform distribution
        :param flag: Random inactivation type (Optional: "Gaussian" or "uniform")
                     when assigned to "Gaussian", the random inactivation probability follows two-dimensional Gaussian distribution
                     (i.e. the random inactivation probability is proportional to the distance from the central neuron);
                     when assigned to "uniform", the random inactivation probability follows uniform distribution between 0 and 1
                     (i.e. the random inactivation probability is the same across whole kernel)
        :param batch_size: batch processing size
        """

        if flag == 'Gaussian':
            # Cook the random inactivation probability matrix to meet the batch processing requirements
            random_inactivation_probability = self.random_inactivation_probability_matrix.repeat(channel, channel, 1, 1,
                                                                                                 1).to(device)

            # Normalize the probability into range 0-1
            random_inactivation_probability = random_inactivation_probability / random_inactivation_probability[
                0, 0, int((dimension - 1) / 2), int((dimension - 1) / 2),
                int((dimension - 1) / 2)]

            # Generate random number between 0-1
            random_number = torch.rand(channel, channel, 1, dimension, dimension).to(device)

            # Random inactivation
            matrix = random_number < random_inactivation_probability

        if flag == 'uniform':
            # Generate random number between 0-1
            random_number = torch.rand(channel, channel, 1, dimension, dimension).to(device)

            # Constant random inactivation probability
            random_inactivation_probability = torch.ones(channel, channel, 1, dimension, dimension, device=device) * P

            # Random inactivation
            matrix = random_number < random_inactivation_probability

        return matrix

    # Methods for RCNNshift tracking
    def wei(self, ):
        """
        Generate the wight matrix of ROI based on the epanechnikov kernel.
        """
        width, height, center = self.search_window()
        wei_width = np.arange(width)
        wei_height = np.arange(height)
        dist_width = (wei_width[:, None] - center[0]) ** 2
        dist_height = (wei_height - center[1]) ** 2
        z = dist_width + dist_height
        wei = 1 - z / (center[0] ** 2 + center[1] ** 2)
        return wei

    def hist_first(self, wei, ignition):
        """
        This method draw the histogram of ROI in the video's first frame.
        :param wei: wight matrix of ROI based on the epanechnikov kernel
        :param ignition: ignition results of the video
        """
        ignition_first = ignition[:, :, 0]
        track_rect = self.track_rect
        track_ROI = self.track_ROI

        ignition_first_frame = np.array(ignition_first, dtype=np.uint8)
        ignition_first_ROI = ignition_first_frame[int(track_rect[1]):int(track_rect[1] + track_rect[3]),
                             int(track_rect[0]):int(track_rect[0] + track_rect[2])]
        (B, G, R) = cv2.split(track_ROI)
        first_ROI = cv2.merge((B, G, R, ignition_first_ROI))

        C = 1 / sum(sum(wei))
        hist_first = np.zeros(int(16 + self.weight * self.t / 2 + 1))
        q_r = np.floor_divide(R, 16).astype(np.uint8)
        q_ignition = np.floor_divide(ignition_first_ROI, int(self.t/8)).astype(np.uint8)
        quantized_features = (q_r + q_ignition * self.weight).astype(np.uint8)
        np.add.at(hist_first, quantized_features.flatten(), wei.flatten())
        hist_first *= C

        return hist_first

    def RCNNshift_track(self, ignition, hist_first, wei):
        """
        This method performs RCNNshift object tracking.
        :param ignition: ignition maps of the to be tracked video
        :param hist_first: histogram of ROI in the video's first frame
        :param wei: wight matrix of ROI based on the epanechnikov kernel
        """
        # Extract the width, height, and center position of the search window
        width, height, center = self.search_window()
        track_rect = copy.copy(self.track_rect)

        # Read video
        cap = cv2.VideoCapture(self.read_video())

        C = 1 / sum(sum(wei))
        m = 0

        # Saving path for tracking window
        if self.select_tracker == 'RCNNshift_3D':
            path = os.path.join('TrackWindowResult', 'RCNNshift_3D', f"{self.name}.txt")
        elif self.select_tracker == 'RCNNshift':
            path = os.path.join('TrackWindowResult', 'RCNNshift', f"{self.name}.txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'a').close()

        # Start tracking
        with open(path, "w") as f:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                img = frame
                num = 0
                Y = [1, 1]

                # Add the RCNN ignition maps as the fourth channel of images
                ignition_current = ignition[:, :, m]
                m += 1

                ignition_current_frame = np.array(ignition_current, dtype=np.uint8)

                (B, G, R) = cv2.split(img)
                current_frame = cv2.merge((B, G, R, ignition_current_frame))

                while (np.sqrt(Y[0] ** 2 + Y[1] ** 2) > 0.5) & (num < 20):
                    num = num + 1
                    current_frame_ROI = current_frame[int(track_rect[1]):int(track_rect[1] + track_rect[3]),
                                        int(track_rect[0]):int(track_rect[0] + track_rect[2])]

                    # Compute histogram of the candidate region
                    hist_current = np.zeros(int(16 + self.weight * self.t / 2 + 1))

                    q_r = np.floor_divide(current_frame_ROI[:, :, 2], 16).astype(np.uint8)
                    q_ignition = np.floor_divide(current_frame_ROI[:, :, 3], int(self.t/8)).astype(np.uint8)
                    quantized_features = (q_r + q_ignition * self.weight).astype(np.uint8)

                    np.add.at(hist_current, quantized_features.flatten(), wei.flatten())
                    hist_current *= C

                    # Compute histogram difference
                    nonzero_indices = hist_current != 0
                    w = np.zeros(int(16 + self.weight * self.t / 2 + 1))
                    w[nonzero_indices] = np.sqrt(hist_first[nonzero_indices] / hist_current[nonzero_indices])

                    # Compute meanshift parameters
                    i, j = np.meshgrid(np.arange(0, width), np.arange(0, height))
                    sum_w = np.sum(w[quantized_features[i, j]])
                    sum_xw = np.sum(w[quantized_features[i, j]] * np.array([i - center[0], j - center[1]]), axis=(1, 2))
                    Y = sum_xw / (sum_w + 1e-6)

                    # Update track window
                    track_rect[0] = track_rect[0] + Y[1]
                    track_rect[1] = track_rect[1] + Y[0]

                    # Prevent the track window from going outside the image
                    if track_rect[0] >= 0 and track_rect[1] >= 0 and track_rect[0] + track_rect[2] <= img.shape[1] \
                            and track_rect[1] + track_rect[3] <= img.shape[0]:
                        continue
                    else:
                        track_rect = copy.copy(self.track_rect)

                # Save track window
                v0, v1, v2, v3 = map(int, track_rect)
                window = (v0, v1, v2, v3)
                window_str = str(window).replace(" ", "").replace("(", "").replace(")", "")
                f.write(window_str + '\r')

                # Display tracking window in the image
                pt1 = (v0, v1)
                pt2 = (v0 + v2, v1 + v3)
                IMG = cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)

                if self.perform == 'live':
                    cv2.imshow('img2', IMG)
                else:
                    pass

                if cv2.waitKey(60) & 0xff == 27:
                    break

        cap.release()

    def search_window(self):
        """
        Extract the width, height, and center position of the search window.
        """
        # Extract the width and height of the search window
        width = self.track_rect[3]
        height = self.track_rect[2]

        # Calculate the center position of the search window
        center = [width / 2, height / 2]

        return width, height, center

    def read_video(self):
        """
        This method is used to read the video file and returns the path of the video.
        """
        file_extension = ".mp4"
        file_path = os.path.join(self.video_path, str(self.name) + file_extension)
        if os.path.exists(file_path):
            # If the file exists, return the file path
            return file_path
        else:
            # If the file does not exist, raise an exception with an error message
            raise FileNotFoundError("Video file does not exist: " + file_path)

    def mouse_select(self, img):
        """
        This method calls the on_mouse method to select and return information about the target ROI.

        :param img: The image on which the target box needs to be selected.
       """
        print("Select the target ROI by\033[1m clicking and dragging the mouse\033[0m. Press\033[1m Enter\033[0m when "
              "the selection is done.")
        mouse_params = {'x': None, 'width': None, 'height': None,
                        'y': None}
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.on_mouse, mouse_params)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return [mouse_params['x'], mouse_params['y'], mouse_params['width'],
                mouse_params['height']]

    @staticmethod
    def on_mouse(event, x, y, flags, param):
        """
        This method allows the user to select a target ROI by clicking the left mouse button to start the selection
        and releasing the mouse button to finalize it. It reads the position of the search box and captures
        the pixel information of the image enclosed by the target ROI.

        :param event: Event triggered by the left mouse button.
        :param x: The leftmost position of the target ROI.
        :param y: The bottom position of the target ROI.
        :param flags: Mouse movement flag.
        :param param: Size and pixel information of the target ROI.
        """
        global point1
        frame = first_frame.copy()

        # When the left mouse button is pressed down
        if event == cv2.EVENT_LBUTTONDOWN:
            point1 = (x, y)
            cv2.circle(frame, point1, 10, (0, 255, 0), 5)
            cv2.imshow('image', frame)

        # When the left mouse button is pressed down and the mouse is being moved simultaneously
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            cv2.rectangle(frame, point1, (x, y), (255, 0, 0), 5)
            cv2.imshow('image', frame)

        # When the left mouse button is released
        elif event == cv2.EVENT_LBUTTONUP:
            point2 = (x, y)
            cv2.rectangle(frame, point1, point2, (0, 0, 255), 5)
            cv2.imshow('image', frame)
            param['x'] = min(point1[0], point2[0])
            param['y'] = min(point1[1], point2[1])
            param['width'] = abs(point1[0] - point2[0])
            param['height'] = abs(point1[1] - point2[1])

    def meanshift(self, ):
        """
        Method for meanshift tracking.
        """
        cap = cv2.VideoCapture(self.read_video())
        track_rect = self.track_rect
        track_ROI = self.track_ROI
        track_ROI_gray = cv2.cvtColor(track_ROI, cv2.COLOR_BGR2GRAY)

        # Calculate histogram based on feature types
        if np.array_equal(track_ROI, cv2.cvtColor(track_ROI_gray, cv2.COLOR_GRAY2BGR)):
            # In gray space
            gray_roi = track_ROI
            roi_hist = cv2.calcHist([gray_roi], [0], None, [255], [0, 255])

        else:

            # In RGB space
            # rgb_roi = track_ROI
            # roi_hist = cv2.calcHist([rgb_roi], [0, 1, 2], None, [16, 16, 16], [0, 255, 0, 255, 0, 255])

            # In HSV space
            hsv_roi = cv2.cvtColor(track_ROI, cv2.COLOR_BGR2HSV)
            roi_hist = cv2.calcHist([hsv_roi], [0, 1, 2], None, [16, 16, 16], [0, 180, 0, 255, 0, 255])

        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # Saving path for tracking window
        path = os.path.join('TrackWindowResult', 'Meanshift', f"{self.name}.txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'a').close()

        with open(path, 'w') as f:
            term = [cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1]
            track_window = track_rect
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                if np.array_equal(track_ROI, cv2.cvtColor(track_ROI_gray, cv2.COLOR_GRAY2BGR)):
                    # In gray space
                    histogram_backprojection = cv2.calcBackProject([frame], [0], roi_hist, [0, 255], 1)

                else:
                    # In RGB space
                    # histogram_backprojection = cv2.calcBackProject([frame], [0, 1, 2], roi_hist,
                    #                                                [0, 255, 0, 255, 0, 255], 1)

                    # In HSV space
                    hsv_frame_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    histogram_backprojection = cv2.calcBackProject([hsv_frame_roi], [0, 1, 2], roi_hist,
                                                                   [0, 180, 0, 255, 0, 255], 1)
                # Apply mean shift algorithm for object tracking
                ret, track_window = cv2.meanShift(histogram_backprojection, track_window, term)
                (x, y, w, h) = track_window

                # Draw rectangle on the frame
                IMG = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)

                # Save track window
                track_window_str = str(track_window).replace(" ", "").replace("(", "").replace(")", "")
                f.write(track_window_str + '\r')

                # Display
                if self.perform == 'live':
                    cv2.imshow('IMG', IMG)
                else:
                    pass
                if cv2.waitKey(100) & 0xff == ord("q"):
                    break

        cv2.destroyAllWindows()
        cap.release()

    def show(self):
        """
        Save traced video and display it with track window marked in red.
        """
        cap = cv2.VideoCapture(self.read_video())
        ret, frame = cap.read()

        if self.select_tracker == "meanshift":
            path = os.path.join('TrackWindowResult', 'Meanshift', str(self.name) + '.txt')
            output_file = os.path.join('TrackedVideo', 'Meanshift', str(self.name) + '.mp4')
        elif self.select_tracker == "RCNNshift_3D":
            path = os.path.join('TrackWindowResult', 'RCNNshift_3D', str(self.name) + '.txt')
            output_file = os.path.join('TrackedVideo', 'RCNNshift_3D', str(self.name) + '.mp4')
        elif self.select_tracker == "RCNNshift":
            path = os.path.join('TrackWindowResult', 'RCNNshift', str(self.name) + '.txt')
            output_file = os.path.join('TrackedVideo', 'RCNNshift', str(self.name) + '.mp4')

        os.makedirs(os.path.dirname(path), exist_ok=True)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if not os.path.exists(path):
            open(path, 'w').close()

        if not os.path.exists(output_file):
            open(output_file, 'w').close()

        with open(path, 'r') as file:
            lines = [line.rstrip() for line in file]

        frame_num = 0

        # Output video FPS
        output_fps = 30.0

        # Output video size
        height, width, _ = frame.shape
        output_size = (width, height)

        # Save video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_file, fourcc, output_fps, output_size)
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            k = self.convert_str_to_int(lines[frame_num])
            v0, v1, v2, v3 = map(int, k)
            track_window = (v0, v1, v2, v3)
            pt1 = (v0, v1)
            pt2 = (v0 + v2, v1 + v3)
            img = cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
            cv2.imshow('IMG', img)
            video_writer.write(img)
            frame_num = frame_num + 1

            if cv2.waitKey(100) & 0xff == ord("q"):
                break

        cv2.destroyAllWindows()
        cap.release()

    @staticmethod
    def convert_str_to_int(string):
        """
        Extract only numbers from the text file.
        :param string: string of the text file
        """
        nums = string.split(',')
        int_nums = [int(float(num.replace(' ', ''))) for num in nums]
        return int_nums
