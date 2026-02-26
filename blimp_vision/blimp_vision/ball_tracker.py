#!/usr/bin/env python3
"""
Module for BallTracker, which selects the optimal target from detection results.
"""

import numpy as np
# from blimp_vision_msgs.msg import Detection
from collections import defaultdict


class BallTracker:
    def __init__(self, width, height):
        """
        Initialize the BallTracker.
        
        :param width: Frame width.
        :param height: Frame height.
        """
        self.current_tracked_id = None
        self.frame_center = (int(width // 2), int(height // 2))
        self.lost_track_threshold = 5  # Frames to wait before declaring track lost.
        self.frames_without_target = 0
        self.goal_cutoff_index = 3.0
        self.track_history = defaultdict(int)  # Track consecutive detections
        self.min_consecutive_detections = 0  # Minimum consecutive detections required
        self.max_history_size = 30  # Maximum size of track history to prevent memory growth

    def calculate_center_distance(self, detection):
        """
        Calculate Euclidean distance from detection center to frame center.
        
        :param detection: A list or array [x, y, w, h].
        :return: Distance as a float.
        """
        return ((detection[0] - self.frame_center[0]) ** 2 +
                (detection[1] - self.frame_center[1]) ** 2) ** 0.5

    def calculate_area(self, detection):
        """
        Calculate the area of the detection bounding box.
        
        :param detection: A list or array [x, y, w, h].
        :return: Area as a float.
        """
        return detection[2] * detection[3]

    def _clean_track_history(self, current_track_ids):
        """
        Clean up track history by removing entries not in current detections.
        
        :param current_track_ids: List of currently detected track IDs.
        """
        # Reset counters for tracks not seen in current frame
        for track_id in list(self.track_history.keys()):
            if track_id not in current_track_ids:
                self.track_history[track_id] = 0
                
        # Remove entries with zero count to manage memory
        if len(self.track_history) > self.max_history_size:
            self.track_history = {k: v for k, v in self.track_history.items() 
                                 if v > 0 or k == self.current_tracked_id}

    def select_target(self, detections, yellow_goal_mode=None):
        """
        Select the optimal target based on detection size and distance to center.
        
        :param detections: YOLO detection results.
        :param yellow_goal_mode: Optional mode to filter detections by goal color.
        :return: A Detection message for the chosen target, or None.
        """
        # No detections available.
        if not detections or detections.boxes.id is None:
            self.frames_without_target += 1
            if self.frames_without_target >= self.lost_track_threshold:
                self.current_tracked_id = None
            return None

        boxes = detections.boxes.xywh.cpu()
        track_ids = detections.boxes.id.int().cpu().tolist()
        
        # Update track history with current detections
        for track_id in track_ids:
            try:
                # Ensure track_id is hashable (convert to int if necessary)
                track_id_key = int(track_id) if not isinstance(track_id, int) else track_id
                self.track_history[track_id_key] += 1
            except (TypeError, ValueError, KeyError) as e:
                # Log or handle the error gracefully, but don't crash
                continue
        
        self._clean_track_history(track_ids)
        
        best_target = None

        # If already tracking an object, try to continue tracking it.
        if self.current_tracked_id is not None:
            if self.current_tracked_id in track_ids:
                self.frames_without_target = 0
                best_target = self.current_tracked_id
                
                # Find which class this tracked object belongs to
                target_index = track_ids.index(self.current_tracked_id)
                original_index = detections.boxes.id.int().cpu().tolist().index(self.current_tracked_id)
                target_class_index = int(detections.boxes.cls.cpu()[original_index].item())
            else:
                self.frames_without_target += 1
                if self.frames_without_target >= self.lost_track_threshold:
                    self.current_tracked_id = None
        else:
            # If not tracking, choose a new target.
            if yellow_goal_mode is not None:
                if yellow_goal_mode:
                    filtered_indices = detections.boxes.cls.cpu() < self.goal_cutoff_index
                    boxes = boxes[filtered_indices]
                    track_ids = [tid for i, tid in enumerate(track_ids) if filtered_indices[i]]
                else:
                    filtered_indices = detections.boxes.cls.cpu() >= self.goal_cutoff_index
                    boxes = boxes[filtered_indices]
                    track_ids = [tid for i, tid in enumerate(track_ids) if filtered_indices[i]]
            else:
                filtered_indices = detections.boxes.cls.cpu() < 2
                boxes = boxes[filtered_indices]
                track_ids = [tid for i, tid in enumerate(track_ids) if filtered_indices[i]]
                
            best_score = float('inf')
            for box, track_id in zip(boxes, track_ids):
                # Skip if not detected consistently
                if self.track_history.get(track_id, 0) < self.min_consecutive_detections:
                    continue
                    
                center_distance = self.calculate_center_distance(box)
                area = self.calculate_area(box)
                if area < 40:  # Ignore very small detections.
                    continue
                score = (center_distance / area) * 100  # Lower score is better.
                if score < best_score:
                    best_score = score
                    best_target = track_id
                    target_index = boxes.tolist().index(box.tolist())

        if best_target is None:
            return None

        self.current_tracked_id = best_target
        self.frames_without_target = 0

        # Find the detection corresponding to the tracked target
        if 'target_class_index' not in locals():
            # We need to find the target's class index
            original_index = detections.boxes.id.int().cpu().tolist().index(self.current_tracked_id)
            target_class_index = int(detections.boxes.cls.cpu()[original_index].item())
        
        detection_msg = Detection()
        detection_msg.obj_class = detections.names[target_class_index]
        
        if 'filtered_indices' in locals() and 'target_index' in locals():
            detection_msg.bbox = boxes[target_index].tolist()
            detection_msg.confidence = detections.boxes.conf[filtered_indices][target_index].cpu().item()
        else:
            # We're in the already-tracking case
            original_index = detections.boxes.id.int().cpu().tolist().index(self.current_tracked_id)
            detection_msg.bbox = detections.boxes.xywh.cpu()[original_index].tolist()
            detection_msg.confidence = detections.boxes.conf.cpu()[original_index].item()
            
        detection_msg.track_id = self.current_tracked_id

        return detection_msg
