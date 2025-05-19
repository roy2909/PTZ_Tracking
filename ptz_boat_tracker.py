#!/usr/bin/env python3
"""
Boat Tracking System

A Python application that uses YOLOv8 for real time boat detection and tracking.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import argparse
from typing import Dict, List, Tuple, Optional, Union, Any


class PIDController:
    """
    PID Controller for smooth tracking adjustments.
    """
    
    def __init__(self, kp: float = 0.5, ki: float = 0.0, kd: float = 0.1):
        """
        Initialize a PID controller with gain parameters.
        
        Args:
            kp: Proportional gain coefficient
            ki: Integral gain coefficient
            kd: Derivative gain coefficient
        """
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.prev_error = 0.0
        self.integral = 0.0
        
    def update(self, error: float, dt: float) -> float:
        """
        Update the PID controller with new error measurement.
        
        Args:
            error: The current error value (distance from target)
            dt: Time delta since last update in seconds
            
        Returns:
            float: Control output value
        """
        # Prevent dt from being too small
        if dt < 0.001:
            dt = 0.001
            
        # Calculate PID components
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        
        # Calculate output
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # Update previous error
        self.prev_error = error
        
        return output
    
    def reset(self) -> None:
        """Reset the controller state."""
        self.prev_error = 0.0
        self.integral = 0.0


class BoatTracker:
    """
    Boat tracking system using YOLOv8 and PID controllers.
    
    This class handles detection, tracking, and zooming in on detected boats.
    """
    
    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        """
        Initialize the boat tracker with model and parameters.
        
        Args:
            model_path: Path to the YOLOv8 model file
            confidence_threshold: Minimum detection confidence threshold
        """
        # Load YOLOv8 model
        self.model = YOLO(model_path)  
        self.confidence_threshold = confidence_threshold
        
        # Initialize PID controllers for x and y coordinates
        self.pid_x = PIDController(kp=0.2, ki=0.0, kd=0.01)
        self.pid_y = PIDController(kp=0.2, ki=0.00, kd=0.01)
        
        # Initialize tracking variables
        self.tracking_enabled = False
        self.tracking_id = None  # ID to keep track of which boat we're tracking
        self.target_bbox: Optional[Tuple[int, int, int, int]] = None  # Store the last known bounding box
        self.zoom_level = 1.0
        self.target_zoom = 1.0
        self.zoom_pid = PIDController(kp=0.3, ki=0.0, kd=0.2)
        self.center_x = 0.0
        self.center_y = 0.0
        self.last_detection_time = 0.0
        self.last_update_time = time.time()
        self.prev_width = None
        self.prev_height = None
        
        # Define class IDs for boats in COCO dataset (used by default YOLOv8)
        # 8: boat
        self.boat_class_ids = [8]  
        
        # Selection mode variables
        self.selection_mode = False
        self.candidates: List[Dict[str, Any]] = []
        self.selection_timeout = 0.0
        
    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run YOLOv8 detection on the frame to find boats.
        
        Args:
            frame: Input image frame as numpy array
            
        Returns:
            List of detection dictionaries with bbox, confidence, and class information
        """
        results = self.model(frame, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Only consider boats with confidence above threshold
                if class_id in self.boat_class_ids and confidence > self.confidence_threshold:
                    detections.append({
                        'id': i,  # Give each detection a unique ID within this frame
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'center': ((x1 + x2) / 2, (y1 + y2) / 2)
                    })
        
        return detections
    
    def find_best_matching_boat(self, detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Find the best matching boat to our current tracking target.
        
        Uses Intersection over Union and center distance to find the most
        likely match for the currently tracked boat.
        
        Args:
            detections: List of detected objects
            
        Returns:
            Best matching detection or None if no good match found
        """
        if not detections or self.target_bbox is None:
            return None
            
        best_iou = 0.0
        best_detection = None
        
        # Calculate center of the tracked object
        tx1, ty1, tx2, ty2 = self.target_bbox
        tracked_center_x = (tx1 + tx2) / 2
        tracked_center_y = (ty1 + ty2) / 2
        
        # Find detection with highest IoU
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # Calculate distance between centers
            center_x, center_y = detection['center']
            distance = np.sqrt((center_x - tracked_center_x)**2 + (center_y - tracked_center_y)**2)
            
            # Calculate IoU 
            intersection_x1 = max(tx1, x1)
            intersection_y1 = max(ty1, y1)
            intersection_x2 = min(tx2, x2)
            intersection_y2 = min(ty2, y2)
            
            if intersection_x1 < intersection_x2 and intersection_y1 < intersection_y2:
                intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                tracked_area = (tx2 - tx1) * (ty2 - ty1)
                detection_area = (x2 - x1) * (y2 - y1)
                union_area = tracked_area + detection_area - intersection_area
                iou = intersection_area / union_area
            else:
                iou = 0.0
                
            # Prefer objects with higher IoU but also consider distance
            combined_score = iou * (1 - min(distance, 100) / 100)
            
            if combined_score > best_iou:
                best_iou = combined_score
                best_detection = detection
        
        # Only accept matches with reasonable IoU
        if best_iou > 0.2:
            return best_detection
        return None
    
    def update_tracking(self, target: Optional[Dict[str, Any]], frame_shape: Tuple[int, ...]) -> bool:
        """
        Update tracking information based on target detection.
        
        
        Args:
            target: The detected target object or None if lost
            frame_shape: Dimensions of the frame (height, width, channels)
            
        Returns:
            bool: Whether tracking is still active
        """
        if target is None:
            if time.time() - self.last_detection_time > 1.0:
                return False
            return self.tracking_enabled

        # Update last detection time
        self.last_detection_time = time.time()

        # Store the current bbox for future matching
        self.target_bbox = target['bbox']

        # Extract bounding box and calculate center
        x1, y1, x2, y2 = target['bbox']
        center_x, center_y = target['center']

        # Calculate frame center
        frame_height, frame_width = frame_shape[:2]
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2

        # Time since last update
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # Error between object center and frame center
        error_x = center_x - frame_center_x
        error_y = center_y - frame_center_y

        # Update PID corrections
        correction_x = self.pid_x.update(error_x, dt)
        correction_y = self.pid_y.update(error_y, dt)

        # Apply smoothing to tracking center
        self.center_x = max(0.0, min(center_x - correction_x * 0.5, frame_width))
        self.center_y = max(0.0, min(center_y - correction_y * 0.5, frame_height))

        # Smooth bounding box width and height
        object_width = x2 - x1
        object_height = y2 - y1

        # Initialize previous values
        if self.prev_width is None:
            self.prev_width = float(object_width)
            self.prev_height = float(object_height)

        # Smooth box dimensions
        smoothing_factor = 0.1
        self.prev_width = (1 - smoothing_factor) * self.prev_width + smoothing_factor * object_width
        self.prev_height = (1 - smoothing_factor) * self.prev_height + smoothing_factor * object_height

        # Compute zoom target based on smoothed dimensions
        width_ratio = frame_width / max(self.prev_width, frame_width * 0.1)
        height_ratio = frame_height / max(self.prev_height, frame_height * 0.1)
        self.target_zoom = min(min(width_ratio, height_ratio) * 0.6, 5.0)

        zoom_error = self.target_zoom - self.zoom_level

        if abs(zoom_error) > 0.05:  # Deadband threshold
            zoom_correction = self.zoom_pid.update(zoom_error, dt)


            alpha = 0.05
            proposed_zoom = self.zoom_level + zoom_correction
            smoothed_zoom = (1 - alpha) * self.zoom_level + alpha * proposed_zoom

            self.zoom_level = max(1.0, min(smoothed_zoom, 5.0))

        return self.tracking_enabled

    
    def apply_zoom(self, frame: np.ndarray) -> np.ndarray:
        """
        
        Creates a zoomed view centered on the tracked boat by cropping
        and resizing the image.
        
        Args:
            frame: Input frame
            
        Returns:
            Zoomed frame
        """
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate crop region
        crop_width = int(frame_width / self.zoom_level)
        crop_height = int(frame_height / self.zoom_level)
        
        # Center crop around tracked object
        start_x = max(0, int(self.center_x - crop_width / 2))
        start_y = max(0, int(self.center_y - crop_height / 2))
        
        # Ensure crop region is within frame bounds
        if start_x + crop_width > frame_width:
            start_x = frame_width - crop_width
        if start_y + crop_height > frame_height:
            start_y = frame_height - crop_height
            
        # Extract region of interest
        roi = frame[start_y:start_y + crop_height, start_x:start_x + crop_width]
        
        # Resize back to original dimensions
        zoomed_frame = cv2.resize(roi, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
        
        return zoomed_frame
    
    def handle_selection_mode(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        
        Creates an interactive UI that allows users to select which boat to track
        when multiple are detected.
        
        Args:
            frame: Input frame to draw UI on
            detections: List of detected boats
            
        Returns:
            Frame with selection UI drawn on it
        """
        if not self.selection_mode:
            if len(detections) > 1 and not self.tracking_enabled:
                self.selection_mode = True
                self.candidates = detections
                self.selection_timeout = time.time() + 10  # 10 seconds to select
                print("Multiple boats detected. Press 1-9 to select which boat to track.")
            return frame
        
        # If in selection mode, check for timeout
        if time.time() > self.selection_timeout:
            self.selection_mode = False
            print("Selection timeout - exiting selection mode.")
            return frame
            
        # Draw numbers on each candidate
        selection_frame = frame.copy()
        for i, candidate in enumerate(self.candidates):
            if i >= 9:  # Only handle up to 9 candidates
                break
                
            x1, y1, x2, y2 = candidate['bbox']
            
            # Draw box with number
            cv2.rectangle(selection_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # Draw selection number
            number_text = str(i + 1)
            text_size = cv2.getTextSize(number_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x = int((x1 + x2) / 2 - text_size[0] / 2)
            text_y = int((y1 + y2) / 2 + text_size[1] / 2)
            
            # Draw background rectangle for text
            cv2.rectangle(selection_frame, 
                         (text_x - 5, text_y - text_size[1] - 5), 
                         (text_x + text_size[0] + 5, text_y + 5), 
                         (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(selection_frame, number_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        # Draw selection instructions
        cv2.putText(selection_frame, "Press 1-9 to select boat to track", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return selection_frame
    
    def draw_tracking_info(self, frame: np.ndarray, target: Optional[Dict[str, Any]]) -> np.ndarray:
        """
        Draw bounding box and tracking information on frame.
        
        Args:
            frame: Input frame to draw on
            target: Currently tracked target or None
            
        Returns:
            Frame with tracking information drawn on it
        """
        # Draw tracking status
        if self.tracking_enabled:
            status_text = "TRACKING ACTIVE"
            status_color = (0, 255, 0)
        else:
            status_text = "TRACKING DISABLED (Press 'T' to enable)"
            status_color = (0, 0, 255)
            
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Draw zoom level
        cv2.putText(frame, f"Zoom: {self.zoom_level:.1f}x", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # If no target or tracking disabled, don't draw bounding box
        if target is None or not self.tracking_enabled:
            return frame
        
        # Draw bounding box
        x1, y1, x2, y2 = target['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw center point of tracking
        cv2.circle(frame, (int(self.center_x), int(self.center_y)), 5, (255, 0, 0), -1)
        
        # Draw info text
        confidence = target['confidence']
        cv2.putText(frame, f"Boat: {confidence:.2f}", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def process_key(self, key: int, detections: List[Dict[str, Any]]) -> None:
        """
        User input handling for toggling tracking, selecting boats, and resetting.
        
        Args:
            key: Key code from cv2.waitKey
            detections: List of current detections
        """
        # Toggle tracking with 't' key
        if key == ord('t') or key == ord('T'):
            self.tracking_enabled = not self.tracking_enabled
            if self.tracking_enabled:
                print("Tracking enabled")
            else:
                print("Tracking disabled")
                
        # Handle selection mode keys (1-9)
        if self.selection_mode and ord('1') <= key <= ord('9'):
            selection_index = key - ord('1')
            if selection_index < len(self.candidates):
                # Select this boat for tracking
                self.target_bbox = self.candidates[selection_index]['bbox']
                self.tracking_id = self.candidates[selection_index]['id']
                self.tracking_enabled = True
                self.selection_mode = False
                print(f"Selected boat {selection_index + 1} for tracking")
                
        # Reset tracking with 'r' key
        if key == ord('r') or key == ord('R'):
            self.tracking_enabled = False
            self.target_bbox = None
            self.tracking_id = None
            self.pid_x.reset()
            self.pid_y.reset()
            self.zoom_pid.reset()
            print("Tracking reset")
            
        # Toggle selection mode with 's' key
        if key == ord('s') or key == ord('S'):
            if not self.selection_mode and len(detections) > 0:
                self.selection_mode = True
                self.candidates = detections
                self.selection_timeout = time.time() + 10
                print("Entering selection mode. Press 1-9 to select boat.")
            else:
                self.selection_mode = False
                print("Exiting selection mode.")
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> None:
        """
        Process a video file for boat tracking.
        
        Args:
            video_path: Path to input video file
            output_path: Optional path to save output video
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Setup output video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
        
        # Reset tracking state
        self.tracking_id = None
        self.target_bbox = None
        self.pid_x.reset()
        self.pid_y.reset()
        self.zoom_pid.reset()
        
        print("Boat Tracking Controls:")
        print("  T - Toggle tracking on/off")
        print("  S - Enter selection mode (when multiple boats detected)")
        print("  1-9 - Select specific boat in selection mode")
        print("  R - Reset tracking")
        print("  Q - Quit")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect boats in the frame
            detections = self.detect_objects(frame)
            
            # Handle selection mode if active
            if self.selection_mode:
                display_frame = self.handle_selection_mode(frame.copy(), detections)
            else:
                display_frame = frame.copy()
            
            # Find target based on tracking state
            target = None
            if self.tracking_enabled and self.target_bbox is not None:
                target = self.find_best_matching_boat(detections)
            
            # Update tracking
            tracking_active = self.update_tracking(target, frame.shape)
            
            # Apply digital zoom if tracking is active
            if tracking_active and self.target_bbox is not None:
                zoomed_frame = self.apply_zoom(frame.copy())
            else:
                zoomed_frame = frame.copy()
            
            # Draw tracking info
            display_frame = self.draw_tracking_info(display_frame, target)
            
            # Side combined display
            combined_frame = np.hstack((display_frame, zoomed_frame))
            
            # Write frame to output video 
            if out:
                out.write(combined_frame)
            
            # Display the frame
            cv2.imshow('Boat Tracking', combined_frame)
            
            # Process key presses
            key = cv2.waitKey(1) & 0xFF
            self.process_key(key, detections)
            
            if key == ord('q'):
                break
        
        # Release resources
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
    
    def start_webcam_tracking(self) -> None:
        """Start live webcam tracking."""
        cap = cv2.VideoCapture(0)  # Use default webcam
        
        # Reset tracking state
        self.tracking_id = None
        self.target_bbox = None
        self.pid_x.reset()
        self.pid_y.reset()
        self.zoom_pid.reset()
        
        print("Boat Tracking Controls:")
        print("  T - Toggle tracking on/off")
        print("  S - Enter selection mode (when multiple boats detected)")
        print("  1-9 - Select specific boat in selection mode")
        print("  R - Reset tracking")
        print("  Q - Quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect boats in the frame
            detections = self.detect_objects(frame)
            
            # Handle selection mode if active
            if self.selection_mode:
                display_frame = self.handle_selection_mode(frame.copy(), detections)
            else:
                display_frame = frame.copy()
            
            # Find target based on tracking state
            target = None
            if self.tracking_enabled and self.target_bbox is not None:
                # Try to find the boat we're currently tracking
                target = self.find_best_matching_boat(detections)
            
            # Update tracking
            tracking_active = self.update_tracking(target, frame.shape)
            
            # Apply digital zoom if tracking is active
            if tracking_active and self.target_bbox is not None:
                zoomed_frame = self.apply_zoom(frame.copy())
            else:
                zoomed_frame = frame.copy()
            
            # Draw tracking info
            display_frame = self.draw_tracking_info(display_frame, target)
            
            # Side combined display
            combined_frame = np.hstack((display_frame, zoomed_frame))
            
            # Resize if too large for screen
            if combined_frame.shape[1] > 1920:
                scale = 1920 / combined_frame.shape[1]
                combined_frame = cv2.resize(combined_frame, 
                                           (int(combined_frame.shape[1] * scale), 
                                            int(combined_frame.shape[0] * scale)))
            
            # Display the frame
            cv2.imshow('Boat Tracking (Webcam)', combined_frame)
            
            # Process key presses
            key = cv2.waitKey(1) & 0xFF
            self.process_key(key, detections)
            
            if key == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Boat Tracking System")
    
    # Input source options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", type=str, help="Path to input video file")
    input_group.add_argument("--webcam", action="store_true", help="Use webcam as input source")
    
    # Output options
    parser.add_argument("--output", type=str, help="Path to save output video (only for video input)")
    
    # Model options
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLOv8 model")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    
    return parser.parse_args()


def main():
    """Main entry point for tracking code."""
    args = parse_args()
    
    tracker = BoatTracker(model_path=args.model, confidence_threshold=args.confidence)
    
    if args.webcam:
        print("Starting webcam tracking...")
        tracker.start_webcam_tracking()
    else:
        print(f"Processing video: {args.video}")
        if args.output:
            print(f"Saving output to: {args.output}")
        tracker.process_video(args.video, args.output)
    
    print("Completed!")


if __name__ == "__main__":
    main()

