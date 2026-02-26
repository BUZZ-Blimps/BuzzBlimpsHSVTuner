import cv2
import numpy as np
from blimp_vision_msgs.msg import Detection

# Define parameters
purple_hsv = {
    "h_min": 110,
    "h_max": 159,
    "s_min": 67,
    "s_max": 255,
    "v_min": 112,
    "v_max": 188
}

# purple_hsv = {
#     "h_min": 116,
#     "h_max": 159,
#     "s_min": 88,
#     "s_max": 255,
#     "v_min": 119,
#     "v_max": 200
# }

def contour_find_ball(left_frame):
    frame = left_frame
    height = frame.shape[0]

    # Convert to HSV
    image_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get HSV bounds from sliders
    hsv_bounds = purple_hsv
    h_min = hsv_bounds["h_min"]
    h_max = hsv_bounds["h_max"]
    s_min = hsv_bounds["s_min"]
    s_max = hsv_bounds["s_max"]
    v_min = hsv_bounds["v_min"]
    v_max = hsv_bounds["v_max"]

    lower_bound = (h_min, s_min, v_min)
    upper_bound = (h_max, s_max, v_max)

    # Create mask and result
    mask = cv2.inRange(image_HSV, lower_bound, upper_bound)
    
    # Dilate mask
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=3)

    score_threshold = 0.2

    # Find contours
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)
    contour_areas = np.zeros((num_contours,1))
    for i in range(num_contours):
        contour_areas[i] = cv2.contourArea(contours[i])
    sorted_indices = np.argsort(contour_areas, axis=0)[::-1]

    for i in range(num_contours):
        contour_index = int(sorted_indices[i])
        contour = contours[contour_index]
        min_area = 450
        contour_score_circle = evaluate_contour_circle_quality(mask, contour, min_area)
        contour_score = contour_score_circle

        if(contour_score >= score_threshold):
            
            # Detected!
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w/2
            cy = y + h/2

            # To prevent len flare (from seeing lights), just ignore detections in top portion of screen
            if cy > height/4:
                detection_msg = Detection()
                detection_msg.class_id = 1
                detection_msg.obj_class = "Shape"
                detection_msg.bbox[0] = cx
                detection_msg.bbox[1] = cy
                detection_msg.bbox[2] = w
                detection_msg.bbox[3] = h
                detection_msg.depth = -1.0
                detection_msg.confidence = -1.0
                detection_msg.track_id = -1

                return detection_msg


def evaluate_contour_circle_quality(mask, contour, min_area=500, max_aspect_ratio=1.5):
    if not has_good_area_and_aspect(contour, min_area, max_aspect_ratio):
        return 0.0

    return fit_circle_and_score(mask, contour)

def has_good_area_and_aspect(contour, min_area=500, max_aspect_ratio=1.5):
    area = cv2.contourArea(contour)
    if area < min_area:
        return False

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = max(w / h, h / w)
    return aspect_ratio <= max_aspect_ratio

def fit_circle_and_score(mask, contour, scale=0.6):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)

    if radius < 5:
        return 0.0  # Skip very small circles

    # Outer and inner masks
    circle_mask = np.zeros_like(mask)

    cv2.circle(circle_mask, center, radius, 255, thickness=-1)

    circle_values = cv2.bitwise_and(mask, mask, mask=circle_mask)

    circle_score = np.count_nonzero(circle_values) / (np.count_nonzero(circle_mask) + 1e-5)

    return max(0.0, circle_score)

def get_centroid(polygon):
    M = cv2.moments(polygon)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy
