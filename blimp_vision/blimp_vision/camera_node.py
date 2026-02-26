#!/usr/bin/env python3
"""
Module defining the CameraNode, which captures images, computes disparity,
runs YOLO detection in parallel, streams the debug view via GStreamer, and
publishes a 3x3 grid of distance values for the entire frame.
"""

import os
import re
import socket
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import yaml
from cv_bridge import CvBridge
from gi.repository import Gst, GLib
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSHistoryPolicy,
                       QoSReliabilityPolicy, QoSDurabilityPolicy)
from std_msgs.msg import Bool, Float64MultiArray, Int64MultiArray
# from blimp_vision_msgs.msg import PerformanceMetrics
# from ultralytics import YOLO

from blimp_vision.ball_tracker import BallTracker
from blimp_vision.contour_goal_detection import contour_find_goal
from blimp_vision.blob_detector import BlobDetectorClass

# Initialize GStreamer once at startup.
Gst.init(None)
cv2.setUseOptimized(True)
cv2.setNumThreads(2)
print(f'OpenCV enabled?: {cv2.ocl.haveOpenCL()}')

class CameraNode(Node):
    """ROS2 Node for vision processing using YOLO and stereo disparity."""

    def __init__(self):
        super().__init__('blimp_vision_node')
        self.running = True

        self._declare_parameters()
        self._get_parameters()

        # Initialize capture device.
        if self.input_video_path == "":
            self.cap = cv2.VideoCapture(self.device_path)
            if not self.cap.isOpened():
                self.get_logger().error(f'Failed to open camera at {self.device_path}')
                return
        else:
            self.cap = cv2.VideoCapture(self.input_video_path)
            if not self.cap.isOpened():
                self.get_logger().error(f'Failed to open camera at {self.input_video_path}')
                return

        self._setup_camera()
        self._load_calibration()
        self._setup_stereorectification()
        self._setup_stereo_matcher()

        # Initialize YOLO models.
        # self.ball_rknn = YOLO(self.ball_model_file, task='detect')
        # self.goal_rknn = YOLO(self.goal_model_file, task='detect')

        # Blob detector
        self.blobDetector = BlobDetectorClass()

        # Other variables.
        self.bridge = CvBridge()
        self.ball_search_mode = True
        self.yellow_goal_mode = False
        self.frame_counter = 0
        self.track_history = defaultdict(list)
        self.tracker = BallTracker(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2,
                                   self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.state = 0

        # Publishers.
        # self.pub_performance = self.create_publisher(PerformanceMetrics, 'performance_metrics', 10)
        self.pub_detections = self.create_publisher(Float64MultiArray, 'targets', 10)
        self.pub_grid = self.create_publisher(Float64MultiArray, 'grid_distances', 10)

        # Subscribers.
        bool_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.state_sub = self.create_subscription(Int64MultiArray, 'state', self.state_callback, 10)
        self.goalcolor_sub = self.create_subscription(Bool, 'goal_color', self.goal_color_callback, bool_qos)

        # Create thread pool for parallel processing.
        self.thread_pool = ThreadPoolExecutor(max_workers=2)

        # Create a timer callback at 15 Hz.
        self.timer = self.create_timer(1 / 15.0, self.camera_callback)

        # Set up GStreamer streaming pipeline.
        # self._setup_gstreamer()

        # Set up Video Recorder if flag enabled.
        self._setup_videosaver()

    def _declare_parameters(self):
        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_number', 4),
                ('device_path', '/dev/video0'),
                ('calibration_path', 'calibration/'),
                ('ball_model_file', 'models/ball.xml'),
                ('goal_model_file', 'models/goal.xml'),
                ('verbose_mode', False),
                ('save_frames', False),
                ('save_location', 'frames/'),
                ('undistort_camera', True),
                ('goal_circle_height', 1.13),
                ('goal_square_height', 1.17),
                ('goal_triangle_height', 1.50),
                ('input_video_path', ""),
            ]
        )

    def _get_parameters(self):
        self.camera_number = self.get_parameter('camera_number').value
        self.device_path = self.get_parameter('device_path').value
        self.calibration_path = self.get_parameter('calibration_path').value
        self.verbose_mode = self.get_parameter('verbose_mode').value
        self.save_frames = self.get_parameter('save_frames').value
        self.save_location = self.get_parameter('save_location').value
        self.undistort_camera = self.get_parameter('undistort_camera').value
        self.goal_circle_height = self.get_parameter('goal_circle_height').value
        self.goal_square_height = self.get_parameter('goal_square_height').value
        self.goal_triangle_height = self.get_parameter('goal_triangle_height').value
        self.input_video_path = self.get_parameter('input_video_path').value

    def _setup_camera(self):
        """Configure the camera properties."""
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _load_calibration(self):
        """Load camera calibration parameters for left and right cameras using the new approach."""
        try:
            # Left camera calibration.
            left_file = f'camera{self.camera_number}_left.yaml'
            left_path = os.path.join(self.calibration_path, left_file)
            with open(left_path, 'r') as file:
                left_data = yaml.safe_load(file)
                self.left_camera_matrix = np.array(left_data['camera_matrix']['data']).reshape(
                    left_data['camera_matrix']['rows'], left_data['camera_matrix']['cols'])
                self.left_distortion_coefficients = np.array(left_data['distortion_coefficients']['data'])
                self.left_rotation_matrix = np.array(left_data['rotation_matrix']['data']).reshape(
                    left_data['rotation_matrix']['rows'], left_data['rotation_matrix']['cols'])
                self.left_translation_vector = np.array(left_data['translation_vector']['data']).reshape(
                    left_data['translation_vector']['rows'], 1)
                self.left_image_size = tuple(left_data['image_size'])

            # Right camera calibration.
            right_file = f'camera{self.camera_number}_right.yaml'
            right_path = os.path.join(self.calibration_path, right_file)
            with open(right_path, 'r') as file:
                right_data = yaml.safe_load(file)
                self.right_camera_matrix = np.array(right_data['camera_matrix']['data']).reshape(
                    right_data['camera_matrix']['rows'], right_data['camera_matrix']['cols'])
                self.right_distortion_coefficients = np.array(right_data['distortion_coefficients']['data'])
                self.right_rotation_matrix = np.array(right_data['rotation_matrix']['data']).reshape(
                    right_data['rotation_matrix']['rows'], right_data['rotation_matrix']['cols'])
                self.right_translation_vector = np.array(right_data['translation_vector']['data']).reshape(
                    right_data['translation_vector']['rows'], 1)
                self.right_image_size = tuple(right_data['image_size'])

            # Use the vertical focal length (the [1,1] element) from the calibration file.
            self.goal_vertical_focal = self.left_camera_matrix[1, 1]

            self.h_fov = 2 * np.arctan(((self.left_image_size[1] / 2) / self.left_camera_matrix[0][0]))
            self.v_fov = 2 * np.arctan(((self.left_image_size[0] / 2) / self.left_camera_matrix[1][1]))

            self.fx = self.left_camera_matrix[0][0]
            self.fy = self.left_camera_matrix[1][1]

            self.cx = self.left_camera_matrix[0][2]
            self.cy = self.left_camera_matrix[1][2]

            self.get_logger().info('Successfully loaded camera calibration files')
        except Exception as e:
            self.get_logger().error(f'Failed to load calibration files: {e}')

    def _setup_stereorectification(self):
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.left_camera_matrix,
            self.left_distortion_coefficients,
            self.right_camera_matrix,
            self.right_distortion_coefficients,
            (640, 480),
            self.right_rotation_matrix,
            self.right_translation_vector,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha = 0
        )

        # Compute rectification maps if undistortion is not desired.
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            self.left_camera_matrix,
            self.left_distortion_coefficients,
            R1,
            P1,
            (640, 480),
            cv2.CV_32FC1)
            
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            self.right_camera_matrix,
            self.right_distortion_coefficients,
            R2,
            P2,
            (640, 480),
            cv2.CV_32FC1)

    def _setup_stereo_matcher(self):
        """Initialize and configure the stereo matcher using SGBM."""
        minDisparity = 0
        numDisparities = 32  # must be divisible by 16
        blockSize = 12
        P1 = 8 * 1 * blockSize**2   # 8 * 25 = 200
        P2 = 32 * 1 * blockSize**2  # 32 * 25 = 800
        disp12MaxDiff = 1
        uniquenessRatio = 0
        speckleWindowSize = 0
        speckleRange = 0
        mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=minDisparity,
            numDisparities=numDisparities,
            blockSize=blockSize,
            P1=P1,
            P2=P2,
            disp12MaxDiff=disp12MaxDiff,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
            mode=mode
        )

    def _setup_gstreamer(self):
        """Configure and launch the GStreamer pipeline for streaming."""
        hostname = socket.gethostname()
        match = re.search(r'(\d+)$', hostname)
        device_num = int(match.group(1)) if match else 0
        port = 5000 + device_num

        pipeline_str = (
            "appsrc name=mysource emit-signals=True is-live=True format=GST_FORMAT_TIME "
            "caps=video/x-raw,format=BGR,width=640,height=480,framerate=12/1 ! "
            "videoconvert ! videoscale ! video/x-raw,format=I420,width=640,height=480 ! "
            "mpph264enc name=hwenc ! "
            "h264parse ! rtph264pay config-interval=1 pt=96 ! "
            "application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96 ! "
            "udpsink host=192.168.0.200 port={port} sync=false"
        ).format(port=port)

        self.get_logger().info("Launching GStreamer pipeline:\n" + pipeline_str)
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsrc = self.pipeline.get_by_name("mysource")
        self.pipeline.set_state(Gst.State.PLAYING)

        self.stream_timestamp = 0.0
        self.frame_duration = 1.0 / 12.0

    def _setup_videosaver(self):
        if self.save_frames:
            # Ensure the save directory exists.
            os.makedirs(self.save_location, exist_ok=True)
            # Capture an initial frame to determine the frame size.
            ret, frame = self.capture_frame()
            if ret:
                # Left frame dimensions: half the width of the full frame.
                # frame_width = frame.shape[1] // 2
                frame_width = frame.shape[1]
                frame_height = frame.shape[0]
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_path = os.path.join(self.save_location, f'{time.time()}.avi')
                self.video_writer = cv2.VideoWriter(video_path, fourcc, 12, (frame_width, frame_height))
                self.get_logger().info(f'Video recording initialized: {video_path}')
            else:
                self.get_logger().error('Failed to capture initial frame for video writer initialization')

    def capture_frame(self):
        """Capture a frame from the camera."""
        return self.cap.read()

    def state_callback(self, msg):
        old_state = self.state

        self.state = msg.data[0]

        self.ball_search_mode = (self.state in [0, 1, 2, 3])

        if old_state is not self.state and self.ball_search_mode:
            self.get_logger().info('Blimp switched from goal search to ball search, Switching vision model...')
        elif old_state is not self.state and not self.ball_search_mode:
            self.get_logger().info('Blimp switched from ball search to goal search, Switching vision model...')

    def goal_color_callback(self, msg):
        self.yellow_goal_mode = msg.data

    def run_model_ball(self, left_frame):
        """
        Run the YOLO detection model on the provided frame.
        
        :param left_frame: Image from the left camera.
        :return: (inference_time, detection results)
        """
        t_start = time.time()
        
        # Only detect class 0 for the ball model
        results = self.ball_rknn.track(
            left_frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            conf=0.65,
            classes=[0]  # Only class 0
        )[0]
        
        return time.time() - t_start, results

    def run_model_goal(self, left_frame):
        """
        Run the YOLO detection model on the provided frame.
        
        :param left_frame: Image from the left camera.
        :return: (inference_time, detection results)
        """
        t_start = time.time()
        
        if self.ball_search_mode:
            selected_model = self.ball_rknn
            # Only detect class 0 for the ball model
            results = selected_model.track(
                left_frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                conf=0.65,
                classes=[0]  # Only class 0
            )[0]
        else:
            selected_model = self.goal_rknn
            # Detect all classes for the goal model (do not use 'classes' argument)
            results = selected_model.track(
                left_frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                conf=0.65
            )[0]
        
        return time.time() - t_start, results


    def compute_disparity(self, left_frame, right_frame):
        """
        Compute disparity between two images using SGBM.
        If undistortion is enabled, the raw frames are remapped using rectification maps.
        
        :return: (computation_time, disparity map)
        """
        t_start = time.time()
        
        if self.undistort_camera:
            # Use rectification maps if undistortion is enabled.
            left_rectified = cv2.remap(left_frame, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
            right_rectified = cv2.remap(right_frame, self.right_map1, self.right_map2, cv2.INTER_LINEAR)
        else:
            left_rectified = left_frame
            right_rectified = right_frame

        left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
        disparity = self.stereo.compute(left_gray, right_gray)
        disparity = disparity.astype(np.float32) / 16.0

        # Publish grid depth values.
        grid_depths = self.compute_grid_depths(disparity)
        self.pub_grid.publish(Float64MultiArray(data=grid_depths))

        return time.time() - t_start, disparity

    def compute_depth(self, disparity_value):
        """
        Compute depth using the stereo camera model.
        The result is converted to meters.
        """
        focal_length = self.left_camera_matrix[0, 0]
        baseline = -self.right_translation_vector[0, 0]
        disparity_value = max(disparity_value, 1e-6)
        depth = ((focal_length * baseline) / disparity_value)
        return depth

    def mono_depth_estimator(self, h, w):
        """
        Estimate depth based on bounding box dimensions using a predefined exponential model.
        """
        a = 2.329
        b = -0.0001485
        c = 0.8616
        d = -0.8479e-05
        
        if self.ball_search_mode:
            small_side = np.min([h, w])
            bbox_area = small_side**2
            return a * np.exp(b * bbox_area) + c * np.exp(d * bbox_area)
        else:
            return np.inf

    def filter_disparity(self, disparity, bbox):
        """
        Filter disparity values within a bounding box and compute the depth.
        Assumes bbox = [center_x, center_y, width, height].
        
        For ball mode: remove inf and nan values, then select only the values within the middle IQR.
        For goal mode: apply a mask (0.6w x 0.6h) centered in the ROI and use those values.
        """
        x, y, w, h = bbox.astype(int)
        x_min = max(x - w // 2, 0)
        x_max = min(x + w // 2, disparity.shape[1])
        y_min = max(y - h // 2, 0)
        y_max = min(y + h // 2, disparity.shape[0])
        roi = disparity[y_min:y_max, x_min:x_max]
        # Filter out inf and nan values
        valid = roi[np.isfinite(roi)]
        if valid.size == 0:
            return float('inf')
        if self.ball_search_mode:
            # Use middle IQR of the valid disparity values.
            q1 = np.percentile(valid, 25)
            q3 = np.percentile(valid, 75)
            filtered = valid[(valid >= q1) & (valid <= q3)]
            if filtered.size == 0:
                filtered = valid
        else:
            # Goal mode: use only the central mask of ROI (0.6w x 0.6h)
            roi_h, roi_w = roi.shape
            center_x = roi_w // 2
            center_y = roi_h // 2
            mask_w = int(0.6 * roi_w)
            mask_h = int(0.6 * roi_h)
            mask_x_min = max(center_x - mask_w // 2, 0)
            mask_x_max = min(center_x + mask_w // 2, roi_w)
            mask_y_min = max(center_y - mask_h // 2, 0)
            mask_y_max = min(center_y + mask_h // 2, roi_h)
            
            roi_mask = np.ones_like(roi, dtype=bool)
            roi_mask[mask_y_min:mask_y_max, mask_x_min:mask_x_max] = False

            # Combine with a finite value check.
            valid_mask = np.isfinite(roi) & roi_mask
            filtered = roi[valid_mask]

        mean_disp = np.mean(filtered)
        return self.compute_depth(mean_disp)

    def compute_grid_depths(self, disparity):
        """
        Divide the disparity map into a 3x3 grid and compute a representative depth (in meters)
        for each region.
        
        :param disparity: The full disparity map.
        :return: A list of 9 depth values.
        """
        h, w = disparity.shape
        grid_depths = []
        cell_h = h // 3
        cell_w = w // 3
        for i in range(3):
            for j in range(3):
                y_min = i * cell_h
                y_max = (i + 1) * cell_h if i < 2 else h
                x_min = j * cell_w
                x_max = (j + 1) * cell_w if j < 2 else w
                region = disparity[y_min:y_max, x_min:x_max]
                if region.size == 0:
                    depth = float('inf')
                else:
                    mean_disp = np.mean(region)
                    depth = self.compute_depth(mean_disp)
                grid_depths.append(depth)
        return grid_depths

    def get_bbox_theta_offsets(self, bbox, depth):
        offset_x = bbox[0] - self.cx
        offset_y = bbox[1] - self.cy
        
        theta_x = np.arctan(offset_x / self.fx)
        theta_y = np.arctan(offset_y / self.fy)
        return theta_x, theta_y

    def camera_callback(self):
        """Main callback: capture, process, stream frames, and publish metrics and grid distances."""
        timing = {}
        t_total = time.time()
        t_preprocess_start = time.time()

        ret, frame = self.capture_frame()
        if not ret:
            self.get_logger().error('Failed to capture frame')
            return

        # Split frame into left and right images.
        frame_width = frame.shape[1] // 2
        left_frame = frame[:, :frame_width]
        right_frame = frame[:, frame_width:]
        
        detection_msg = None

        if self.ball_search_mode:
            detected = False

            # Try contour detection
            contour_detection_msg =  self.blobDetector.contour_find_ball(left_frame)
            if not detected and contour_detection_msg is not None:
                timing['preprocessing'] = 0.0
                timing['disparity'] = 0.0
                # timing['yolo_inference'] = 0.0

                theta_x, theta_y = self.get_bbox_theta_offsets(contour_detection_msg.bbox, contour_detection_msg.depth)
                self.pub_detections.publish(Float64MultiArray(data=[
                    contour_detection_msg.bbox[0],
                    contour_detection_msg.bbox[1],
                    contour_detection_msg.depth,
                    contour_detection_msg.track_id * 1.0,
                    (not self.ball_search_mode) * 1.0,
                    theta_x, theta_y,
                    contour_detection_msg.bbox[2], contour_detection_msg.bbox[3]
                ]))
                detected = True
                detection_msg = contour_detection_msg

            # if not detected:
            #     # Search for balls using YOLO model
        
            #     timing['preprocessing'] = (time.time() - t_preprocess_start) * 1000
            #     disp_future = self.thread_pool.submit(self.compute_disparity, left_frame, right_frame)
            # 
            #     #yolo_future = self.thread_pool.submit(self.run_model, left_frame)
            #     #t_disp, disparity = self.compute_disparity(left_frame, right_frame)
            #     t_yolo, detections = self.run_model_ball(left_frame)

            #     detection_msg = self.tracker.select_target(
            #         detections,
            #         yellow_goal_mode=None if self.ball_search_mode else self.yellow_goal_mode
            #     )

            #     t_disp, disparity = disp_future.result()

            #     timing['disparity'] = t_disp * 1000
            #     timing['yolo_inference'] = t_yolo * 1000

            #     # Process detections.
            #     if detection_msg is not None:
            #         disp_depth = self.filter_disparity(disparity, detection_msg.bbox)
            #         regr_depth = self.mono_depth_estimator(detection_msg.bbox[2], detection_msg.bbox[3])
            #         detection_msg.depth = np.min([disp_depth, regr_depth]) if (np.square(np.max([detection_msg.bbox[2], detection_msg.bbox[3]])) > 900.0) else 100.0

            #         theta_x, theta_y = self.get_bbox_theta_offsets(detection_msg.bbox, detection_msg.depth)

            #         self.pub_detections.publish(Float64MultiArray(data=[
            #             detection_msg.bbox[0],
            #             detection_msg.bbox[1],
            #             detection_msg.depth,
            #             detection_msg.track_id * 1.0,
            #             (not self.ball_search_mode) * 1.0,
            #             theta_x, theta_y,
            #             detection_msg.bbox[2], detection_msg.bbox[3]
            #         ]))
            #         detection = True

            if not detected:
                #nothing founded
                timing['preprocessing'] = 0.0
                timing['disparity'] = 0.0
                # timing['yolo_inference'] = 0.0
                self.pub_detections.publish(Float64MultiArray(data=[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]))

        else:
            # Search for goals
            detected = False

            # Try using contour detection
            contour_detection_msg = contour_find_goal(left_frame, self.yellow_goal_mode)
            if not detected and contour_detection_msg is not None:
                timing['preprocessing'] = 0.0
                timing['disparity'] = 0.0
                timing['yolo_inference'] = 0.0
                bbox_area = contour_detection_msg.bbox[2]*contour_detection_msg.bbox[3]
                if bbox_area >= 8000: # pixel area to start trusting contours over YOLO
                    # Trust contour detection
                    if self.tracker.current_tracked_id is not None:
                        contour_detection_msg.track_id = self.tracker.current_tracked_id
                    theta_x, theta_y = self.get_bbox_theta_offsets(contour_detection_msg.bbox, contour_detection_msg.depth)
                    self.pub_detections.publish(Float64MultiArray(data=[
                        contour_detection_msg.bbox[0],
                        contour_detection_msg.bbox[1],
                        contour_detection_msg.depth,
                        contour_detection_msg.track_id * 1.0,
                        (not self.ball_search_mode) * 1.0,
                        theta_x, theta_y,
                        contour_detection_msg.bbox[2], contour_detection_msg.bbox[3]
                    ]))
                    detected = True
                    detection_msg = contour_detection_msg
            
            # Try using YOLO
            # if not detected:
            #     timing['preprocessing'] = (time.time() - t_preprocess_start) * 1000
            #     disp_future = self.thread_pool.submit(self.compute_disparity, left_frame, right_frame)
            # 
            #     #yolo_future = self.thread_pool.submit(self.run_model, left_frame)
            #     #t_disp, disparity = self.compute_disparity(left_frame, right_frame)
            #     # t_yolo, detections = self.run_model_goal(left_frame)

            #     detection_msg = self.tracker.select_target(
            #         detections,
            #         yellow_goal_mode=None if self.ball_search_mode else self.yellow_goal_mode
            #     )

            #     t_disp, disparity = disp_future.result()

            #     timing['disparity'] = t_disp * 1000
            #     # timing['yolo_inference'] = t_yolo * 1000

            #     # Process detections.
            #     if detection_msg is not None:
            #         # Goal detection mode: use the calibrated vertical focal length and real goal height.
            #         # Assume detection_msg.obj_class contains a string like "circle", "square", or "triangle"
            #         obj_class = detection_msg.obj_class.lower()
            #         if "circle" in obj_class:
            #             real_height = self.goal_circle_height
            #         elif "triangle" in obj_class:
            #             real_height = self.goal_triangle_height
            #         elif "square" in obj_class:
            #             real_height = self.goal_square_height
            #         raw_goal_height = (self.goal_vertical_focal * real_height) / detection_msg.bbox[3]
            #         raw_depth = (self.goal_vertical_focal * real_height) / detection_msg.bbox[3]
            #         # Use the goal_vertical_focal loaded from calibration.
            #         detection_msg.depth = 0.31804 * np.exp(0.59 * raw_depth) + 1.424
            #         
            #         theta_x, theta_y = self.get_bbox_theta_offsets(detection_msg.bbox, detection_msg.depth)

            #         self.pub_detections.publish(Float64MultiArray(data=[
            #             detection_msg.bbox[0],
            #             detection_msg.bbox[1],
            #             detection_msg.depth,
            #             detection_msg.track_id * 1.0,
            #             (not self.ball_search_mode) * 1.0,
            #             theta_x, theta_y,
            #             detection_msg.bbox[2], detection_msg.bbox[3]
            #         ]))

            #         detected = True
            
            # Go back to coutour
            if not detected and contour_detection_msg is not None:
                timing['preprocessing'] = 0.0
                timing['disparity'] = 0.0
                timing['yolo_inference'] = 0.0
                # Trust contour detection
                if self.tracker.current_tracked_id is not None:
                    contour_detection_msg.track_id = self.tracker.current_tracked_id
                theta_x, theta_y = self.get_bbox_theta_offsets(contour_detection_msg.bbox, contour_detection_msg.depth)
                self.pub_detections.publish(Float64MultiArray(data=[
                    contour_detection_msg.bbox[0],
                    contour_detection_msg.bbox[1],
                    contour_detection_msg.depth,
                    contour_detection_msg.track_id * 1.0,
                    (not self.ball_search_mode) * 1.0,
                    theta_x, theta_y,
                    contour_detection_msg.bbox[2], contour_detection_msg.bbox[3]
                ]))
                detected = True
                detection_msg = contour_detection_msg
            
            if not detected:
                self.pub_detections.publish(Float64MultiArray(data=[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]))

        # Prepare debug view.
        debug_view = left_frame.copy()
        img_h, img_w = debug_view.shape[:2]
        center_img = (img_w // 2, img_h // 2)
        # Draw horizontal line
        cv2.line(debug_view, (center_img[0] - 10, center_img[1]), (center_img[0] + 10, center_img[1]), (0, 0, 255), 2)
        # Draw vertical line
        cv2.line(debug_view, (center_img[0], center_img[1] - 10), (center_img[0], center_img[1] + 10), (0, 0, 255), 2)

        if detection_msg is not None:
            x, y, w, h = detection_msg.bbox.astype(int)
            # Define the bounding box corners (assuming x,y is the center of the box)
            top_left = (x - w // 2, y - h // 2)
            bottom_right = (x + w // 2, y + h // 2)
            cv2.rectangle(debug_view, top_left, bottom_right, (0, 255, 0), 2)
            
            # Draw a marker (filled circle) at the center of the bounding box
            cv2.circle(debug_view, (x, y), 3, (255, 0, 0), -1)
            
            cv2.putText(debug_view,
                        f'{detection_msg.obj_class} {detection_msg.track_id} {detection_msg.depth:.1f}m',
                        (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        processed = cv2.resize(debug_view, (640, 480))
        cv2.imshow('debug view', processed)

        if cv2.waitKey(1) == ord('q'):
            self.running = False
        # Stream the processed frame.
        # processed = cv2.resize(debug_view, (640, 480))
        # data = processed.tobytes()
        # buf = Gst.Buffer.new_allocate(None, len(data), None)
        # buf.fill(0, data)
        # buf.pts = int(self.stream_timestamp * Gst.SECOND)
        # buf.duration = int(self.frame_duration * Gst.SECOND)
        # self.stream_timestamp += self.frame_duration
        # flow_return = self.appsrc.emit("push-buffer", buf)
        # if flow_return != Gst.FlowReturn.OK:
        #     self.get_logger().error("Error pushing buffer: " + str(flow_return))

        timing['total'] = (time.time() - t_total) * 1000

        # Publish performance metrics.
        # perf_msg = PerformanceMetrics()
        # perf_msg.yolo_time = timing['yolo_inference']
        # perf_msg.disparity_time = timing['disparity']
        # perf_msg.total_time = timing['total']
        # perf_msg.fps = (1 / perf_msg.total_time) * 1000
        # self.pub_performance.publish(perf_msg)

        if self.save_frames:
            # self.video_writer.write(left_frame)
            self.video_writer.write(frame)

def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = CameraNode()
    try:
        while node.running:
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.save_frames:
            node.video_writer.release()
        node.thread_pool.shutdown(wait=True)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
