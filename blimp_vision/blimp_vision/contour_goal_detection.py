import cv2
import numpy as np

try:
    from blimp_vision_msgs.msg import Detection
except Exception:
    class Detection:
        """Fallback message for non-ROS local testing."""
        def __init__(self):
            self.class_id = 0
            self.obj_class = ""
            self.bbox = [0.0, 0.0, 0.0, 0.0]
            self.depth = 0.0
            self.confidence = 0.0
            self.track_id = 0

# Define parameters
orange_hsv = {
    "h_min": 0,
    "h_max": 25,
    "s_min": 124,
    "s_max": 255,
    "v_min": 240,
    "v_max": 255
}

yellow_hsv = {
    "h_min": 27,
    "h_max": 37,
    "s_min": 107,
    "s_max": 255,
    "v_min": 197,
    "v_max": 255
}

score_threshold = 0.2
polygon_N = 4


## Goal detection function
# Returns None if no detection and Detection if detection
def contour_find_goal(left_frame, yellow_goal_mode):
    detections = contour_find_goals(left_frame, yellow_goal_mode, max_detections=1)
    return detections[0] if detections else None


def contour_find_goals(left_frame, yellow_goal_mode, max_detections=5):
    frame = left_frame

    # Convert to HSV
    image_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get HSV bounds
    if yellow_goal_mode:
        hsv_bounds = yellow_hsv
    else:
        hsv_bounds = orange_hsv

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

    # Find contours
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)
    contour_areas = np.zeros((num_contours,1))
    for i in range(num_contours):
        contour_areas[i] = cv2.contourArea(contours[i])
    sorted_indices = np.argsort(contour_areas, axis=0)[::-1]

    detections = []
    for i in range(num_contours):
        contour_index = int(sorted_indices[i])
        contour = contours[contour_index]
        contour_score_poly = evaluate_contour_quality(mask, contour, polygon_N)
        contour_score_circle = evaluate_contour_circle_quality(mask, contour)
        contour_score = max(contour_score_poly, contour_score_circle)

        if(contour_score >= score_threshold):

            # Detected!
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w/2
            cy = y + h/2

            detection_msg = Detection()
            detection_msg.class_id = 1
            detection_msg.obj_class = "Shape"
            detection_msg.bbox[0] = cx
            detection_msg.bbox[1] = cy
            detection_msg.bbox[2] = w
            detection_msg.bbox[3] = h
            detection_msg.depth = -1.0
            detection_msg.confidence = float(contour_score)
            detection_msg.track_id = -1

            detections.append((cv2.contourArea(contour), detection_msg))

    detections.sort(key=lambda item: item[0], reverse=True)
    return [d for _, d in detections[:max(1, max_detections)]]


## Helper functions
def has_good_area_and_aspect(contour, min_area=500, max_aspect_ratio=1.5):
    area = cv2.contourArea(contour)
    if area < min_area:
        return False

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = max(w / h, h / w)
    return aspect_ratio <= max_aspect_ratio

def approximate_polygon(contour, N):
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == N:
        return approx

    hull = cv2.convexHull(contour)
    if len(hull) < N:
        return None

    indices = np.linspace(0, len(hull)-1, N, dtype=int)
    return hull[indices]

def get_centroid(polygon):
    M = cv2.moments(polygon)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy

def scale_polygon(polygon, center, scale):
    cx, cy = center
    scaled = []
    for pt in polygon:
        x, y = pt[0]
        x_scaled = int(cx + scale * (x - cx))
        y_scaled = int(cy + scale * (y - cy))
        scaled.append([[x_scaled, y_scaled]])
    return np.array(scaled, dtype=np.int32)

def score_polygon_regions(mask, outer_polygon, inner_polygon):
    ring_mask = np.zeros_like(mask)
    inner_mask = np.zeros_like(mask)

    cv2.fillPoly(ring_mask, [outer_polygon], 255)
    cv2.fillPoly(inner_mask, [inner_polygon], 255)

    ring_only = cv2.subtract(ring_mask, inner_mask)

    ring_values = cv2.bitwise_and(mask, mask, mask=ring_only)
    inner_values = cv2.bitwise_and(mask, mask, mask=inner_mask)

    ring_score = np.count_nonzero(ring_values) / (np.count_nonzero(ring_only) + 1e-5)
    inner_penalty = np.count_nonzero(inner_values) / (np.count_nonzero(inner_mask) + 1e-5)

    return max(0.0, ring_score - inner_penalty)

def evaluate_contour_quality(mask, contour, N, min_area=500, max_aspect_ratio=1.5):
    if not has_good_area_and_aspect(contour, min_area, max_aspect_ratio):
        return 0.0

    polygon = approximate_polygon(contour, N)
    if polygon is None or len(polygon) != N:
        return 0.0

    centroid = get_centroid(polygon)
    if centroid is None:
        return 0.0

    inner_polygon = scale_polygon(polygon, centroid, 0.6)
    score = score_polygon_regions(mask, polygon, inner_polygon)

    return score

def fit_circle_and_score(mask, contour, scale=0.6):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)

    if radius < 5:
        return 0.0  # Skip very small circles

    # Outer and inner masks
    outer_mask = np.zeros_like(mask)
    inner_mask = np.zeros_like(mask)

    cv2.circle(outer_mask, center, radius, 255, thickness=-1)
    cv2.circle(inner_mask, center, int(radius * scale), 255, thickness=-1)

    ring_only = cv2.subtract(outer_mask, inner_mask)

    ring_values = cv2.bitwise_and(mask, mask, mask=ring_only)
    inner_values = cv2.bitwise_and(mask, mask, mask=inner_mask)

    ring_score = np.count_nonzero(ring_values) / (np.count_nonzero(ring_only) + 1e-5)
    inner_penalty = np.count_nonzero(inner_values) / (np.count_nonzero(inner_mask) + 1e-5)

    return max(0.0, ring_score - inner_penalty)

def evaluate_contour_circle_quality(mask, contour, min_area=500, max_aspect_ratio=1.5, scale=0.6):
    if not has_good_area_and_aspect(contour, min_area, max_aspect_ratio):
        return 0.0
    return fit_circle_and_score(mask, contour, scale)
