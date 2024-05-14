import time
import math
import numpy as np
import cv2
import pyzed.sl as sl
import torch
import yolov7
from datetime import datetime, timedelta
import rospy
from sensor_msgs.msg import NavSatFix, Image, CameraInfo
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist, TwistStamped
from mavros_msgs.msg import Waypoint, WaypointReached, WaypointList, State, GlobalPositionTarget
from mavros_msgs.srv import WaypointPush, WaypointClear, CommandBool, CommandBoolRequest, SetMode, SetModeRequest


class DroneControl:

    # Constants - Tune if needed!
    GPS_round = 6
    max_gate_width = 3.0  # Maximum relative distance between buoys
    gate_exit_clearance = 0.5  # meters to get clear of buoy gate
    speed_gate_waypoint_clearance = 1.5  # meters to set waypoint from yellow buoy
    search_speed = 2.0  # if no buoys detected move x_meters to look for buoys
    max_searches = 3  # maximum number of times to search for buoys
    look_degrees = 45  # Angle for looking to both sides in degrees

    def __init__(self):

        # Importing External resources
        self.communication = AutoPilotCommunication()
        if torch.cuda.is_available():
            self.device = torch.device(0)
        else:
            self.device = torch.device("cpu")
        print(f"Device: {self.device}")
        self.model = yolov7.load('/home/user/yolo/best.pt')
        self.model.conf = 0.4
        self.model.iou = 0.45

        # Camera setup
        self.DEBUG_CAM = False
        self.zed = sl.Camera()

        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        init_params.coordinate_units = sl.UNIT.METER
        init_params.sdk_verbose = 1
        self.runtime_params = sl.RuntimeParameters()
        self.zed_status = self.zed.open(init_params)

        if self.zed_status != sl.ERROR_CODE.SUCCESS:
            rospy.logerr(f"Failed to initialize camera with error: {repr(self.zed_status)}")
            raise Exception(f"Camera initialization failed: {repr(self.zed_status)}")

        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, -1)
        self.enable_positional_tracking(self.zed)
        rospy.loginfo("Camera initialized!")

        # Declaring variables
        self.sg_no_buoy_dist = None
        self.second_timer = None
        self.docking_timer = None
        self.start_docking = False
        self.start_undocking = False
        self.yellow_set = None
        self.gate_set = False
        self.object_type_list, self.depth_list, self.bearing_list = [], [], []
        self.first_timer = None
        self.docking_hdg = None
        self.start_lon_out_ret = None
        self.start_lat_out_ret = None
        self.start_lon = None
        self.yellow_buoy_lat_3 = None
        self.yellow_buoy_lat_2 = None
        self.yellow_buoy_lat_1 = None
        self.yellow_buoy_lon_3 = None
        self.yellow_buoy_lon_2 = None
        self.yellow_buoy_lon_1 = None
        self.second_is_none = None
        self.closest_color = None
        self.object_GPS = None
        self.wp_yellow_buoy_lon = None
        self.wp_yellow_buoy_lat = None
        self.waypoint_longitude_out_return = None
        self.waypoint_latitude_out_return = None
        self.closest_bearing = None
        self.closest_dist = None
        self.GPS_round = None
        self.second_object_coordinates = None
        self.width = 640
        self.h_fov = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.h_fov
        self.center_x = 640/2
        self.lamda_x = self.h_fov/self.width
        self.waypoint_longitude_out = None
        self.waypoint_latitude_out = None
        self.start_lat = None
        self.buoys_bearing = None
        self.waypoint_longitude = None
        self.waypoint_latitude = None
        self.start_time = None
        self.search_iter = None
        self.port_clear = None
        self.steerboard_clear = None
        self.depth_is_nan = None
        self.depth_img = None
        self.depth_map = None
        self.np_img = None
        self.img = None

    def speed_gate_mission(self):
        if self.gate_set and not self.yellow_set and not self.communication.wp_set:
            print("gate_set and not yellow_set and not wp_set")
            self.process_detection_results()
            if self.get_nth_object(1)['object_type'] == "yellow_buoy":
                self.setup_speed_gate_course()
            else:
                self.advance_towards_next_detection()

        elif not self.gate_set:
            print("not gate_set")
            self.process_detection_results()
            if self.check_buoy_gate():
                self.setup_initial_gate_passage()

        else:
            rospy.loginfo("Navigating to waypoint.")
            time.sleep(0.2)
            
    def docking_mission(self):  # TODO: add switch to LOITER when docked, and GUIDED when leaving
        if self.start_docking:  # Pilot the drone to the docking waypoint while keeping the heading aligned with the buoys
            if self.needs_heading_correction():
                self.correct_heading_towards_docking()
            elif self.timer_reached() and self.first_timer:
                rospy.loginfo("Docking initiated.")
                self.start_timer(35)  # Adjust the time as needed for docking duration
                self.first_timer = False
                self.second_timer = True
            elif self.timer_reached() and self.second_timer:
                self.start_docking = False
                self.start_undocking = True
            else:
                rospy.loginfo("Docking in progress...")
                rospy.sleep(0.2)
        elif self.communication.sub_wp_reached:  # Wait until waypoint in front of docking location has been reached Set next waypoint once it has been.
            rospy.loginfo("Preparing for docking.")
            self.communication.send_guided_wp(self.waypoint_latitude, self.waypoint_longitude)
            self.start_docking = True
            self.start_timer(5)
            self.first_timer = True
        elif self.start_undocking:  # Undocking procedure #TODO: consider changing sleep(5) to something else
            rospy.loginfo("Undocking...")
            self.send_guided_waypoint(self.waypoint_latitude_out_return, self.waypoint_longitude_out_return)
            rospy.sleep(5)
            rospy.loginfo("Undocking complete, Return to Launch called.")
            self.start_undocking = False
            self.communication.RTL()
        else:
            self.initiate_docking_procedure()  # Looks for gate and gets both needed waypoints. Only sets the waypoint in front of the gate

    def nav_channel_mission(self):
        if self.communication.wp_set:  # navigate to waypoint
            if self.communication.waypoint_reached():
                self.communication.wp_set = False
            else:
                rospy.loginfo("Navigating to waypoint.")
                time.sleep(0.1)
            return

        self.process_detection_results()  # If drone is not navigating to a waypoint, look at surroundings

        if self.depth_is_nan:  # In case of bad readings, retry without moving.
            rospy.logerr("Depth is NaN, scanning environment again.")
            return

        closest_object = self.get_nth_object(1)
        if closest_object is not None and closest_object["object_type"] == "yellow_buoy":
            self.handle_yellow_buoy()  # Sets a waypoint on one of the sides of the buoy, determined by the drones heading.

        elif not self.detect_and_traverse_gates():  # Looks for gates and sets waypoints accordingly.
            self.handle_no_detections()  # If no gates or yellow buoys are found, looks left and right, before moving forwards. repeats several times.

    def ship_mission(self):
        # Process detection results to see surroundings
        self.process_detection_results()

        if self.depth_is_nan:  # Retry in case of bad readings
            rospy.logerr("Depth is NaN, scanning environment again.")
            return

        # Detect and handle "other" objects according to COLREGS
        closest_object = self.get_nth_object(1)
        if closest_object is not None and closest_object["object_type"] == "other":
            if self.should_avoid(closest_object):
                self.avoid_object(closest_object)
            else:
                rospy.loginfo("object detected but no avoidance needed.")
        else:
            rospy.loginfo("No other-class objects detected.")

        if self.communication.wp_set:  # If no other type object is detected, and a waypoint is set, navigate there.
            time.sleep(0.1)
            return

        # Detect and traverse gates if no avoidance action is required
        if not self.detect_and_traverse_gates():  # Looks for gates and sets waypoints accordingly.
            self.handle_no_detections()  # If no gates or "other" objects are found, looks left and right, before moving forwards. repeats several times.

    def should_avoid(self, object):
        if object['depth'] < 5:
            return True
        if object['bearing'] < 0:
            return True
        return False
    def avoid_object(self, object):
        rospy.loginfo(f"Avoiding {object['object_type']} at {object['depth']} meters")
        self.communication.clear_waypoints()
        if object['depth'] < 5:
            self.communication.stop()
            return  # Stop the drone if object is too close

        # Calculate new waypoint to avoid collision
        # turn 45 degrees to the right, should avoid collisions with anything on the right while ignoring anything on the left.
        result = self.convert_distance_to_gps_coordinates(2, 45, self.communication.lat, self.communication.lon)
        self.communication.send_guided_wp(result[0], result[1])

    def enable_positional_tracking(self, zed):
        positional_tracking_parameters = sl.PositionalTrackingParameters()
        positional_tracking_parameters.set_floor_as_origin = True
        zed.enable_positional_tracking(positional_tracking_parameters)

    def enable_object_detection(self, zed):
        obj_param = sl.ObjectDetectionParameters()
        obj_param.detection_model = sl.DETECTION_MODEL.CUSTOM_BOX_OBJECTS
        obj_param.enable_tracking = True
        obj_param.enable_mask_output = True
        zed.enable_object_detection(obj_param)

    def get_image_and_depth_map(self):
        rospy.logdebug("Function called: get_image_and_depth_map")
        if self.zed.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
            rospy.logfatal("Failed to grab image and depth map from ZED camera.")
            return False

        self.img = sl.Mat()
        # getting the left picture, we only need one image for object detection
        self.zed.retrieve_image(self.img, sl.VIEW.LEFT)
        self.np_img = self.img.get_data()  # [:, :, :3]  # Convert to numpy array and remove alpha channel.

        # Resize the RGB image to 640x640 pixels using linear interpolation
        self.np_img = cv2.resize(self.np_img, (640, 640), interpolation=cv2.INTER_LINEAR)
        self.np_img = cv2.cvtColor(self.np_img, cv2.COLOR_RGBA2RGB)
        self.depth_map = sl.Mat()
        self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
        self.depth_img = self.depth_map.get_data()

        # Resize the depth map to 640x640 pixels using nearest neighbor interpolation
        self.depth_img = cv2.resize(self.depth_img, (640, 640), interpolation=cv2.INTER_NEAREST)

        return True

    def perform_detections(self):
        rospy.logdebug("Function called: perform_detections")
        print("data type:", type(self.np_img))
        print("Shape:", self.np_img.shape)
        if self.np_img is None:
            rospy.logerror("Image data is not available for detections.")
            return None
        return self.model(self.np_img)

    def process_detection_results(self):
        """Updates object_type_list, depth_list and bearing_list"""
        rospy.logdebug("Function called: process_detection_results")
        object_types = ['red_buoy', 'green_buoy', 'yellow_buoy', 'other_buoy', 'other']
        self.object_type_list, self.depth_list, self.bearing_list = [], [], []
        self.depth_is_nan = False

        if not self.get_image_and_depth_map():
            return

        results = self.perform_detections()
        if results is None:
            rospy.logerror("No detections were made.")
            return

        rospy.logdebug("Detections sucessfully made, Proceeding to processing")

        predictions = results.pred[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        categories = predictions[:, 5]
            
        for i in range(len(predictions)):
            box = boxes[i]
            score = scores[i]
            category = int(categories[i].item())

            box_xyxy = [box[0], box[1], box[2], box[3]]

            center = self.calculate_center(box_xyxy)
            depth = self.get_depth_at_center(center)
            if math.isnan(depth):
                depth = self.get_average_depth_around_center(center, 5)  # Try to get depth by averaging area around center
                if math.isnan(depth):
                    self.depth_is_nan = True
                    continue
            self.depth_list.append(depth)
            self.bearing_list.append(self.calculate_bearing(center))
            self.object_type_list.append(object_types[category])

        rospy.logdebug(f"Object type: {self.object_type_list}, Depths: {self.depth_list}, Bearings: {self.bearing_list}")

    def calculate_center(self, box):
        x_center = (box[2].item() - box[0].item()) / 2 + box[0].item()
        y_center = (box[3].item() - box[1].item()) / 2 + box[1].item()
        return x_center, y_center

    def get_depth_at_center(self, center):
        return self.depth_img[int(center[1])][int(center[0])]

    def get_average_depth_around_center(self, center, size):
        """Calculate the average depth around a specified center within the depth map, ignoring NaNs."""
        x_center, y_center = int(center[0]), int(center[1])
        depths = []

        for y in range(y_center - size // 2, y_center + size // 2 + 1):
            for x in range(x_center - size // 2, x_center + size // 2 + 1):
                depth = self.depth_img[y][x]
                if not math.isnan(depth):
                    depths.append(depth)

        if not depths:
            return float('nan')  # Return NaN if all depths in the region are NaN
        return sum(depths) / len(depths)  # Average of non-NaN depths

    def calculate_bearing(self, center):
        Tx = int(center[0]) - self.center_x
        return Tx * self.lamda_x

    def get_colors_from_detections(self, detection_classes):
        return [self.model.names[int(d_cls)] for d_cls in detection_classes]

    def get_nth_object(self, n):
        """ returns dictionary containing color, distance and bearing of the n-th object, if it doesnt exist returns none"""
        rospy.logdebug("Function called: get_nth_object")

        if n > len(self.depth_list):
            rospy.logdebug(f"Not enough objects detected to find the {n}-th closest object.")
            return None
        rospy.logdebug(f"Getting {n}-th closest object")

        # Create a sorted list of tuples (depth, index) to keep track of original indices
        depth_with_indices = sorted((depth, i) for i, depth in enumerate(self.depth_list))

        try:
            # Get the depth and original index of the n-th closest object
            _, index = depth_with_indices[n - 1]
            distance = round(self.depth_list[index], 2)
            object_type = self.object_type_list[index]
            bearing = round(self.bearing_list[index], 2)

            rospy.logdebug(f"Object {n}: {object_type} at {distance}m, bearing: {bearing} degrees.")

            return {
                "object_type": object_type,
                "distance": distance,
                "bearing": bearing,
            }
        except IndexError:
            rospy.logdebug(f"Failed to get details of the {n}-th closest object.")
            return None

    def check_buoy_gate(self):
        rospy.logdebug("Function called: check_buoy_gate")

        closest_object = self.get_nth_object(1)
        second_closest_object = self.get_nth_object(2)

        if not closest_object or not second_closest_object:
            rospy.logdebug("Insufficient data to determine a gate")
            return False

        if (closest_object['object_type'] == 'green_buoy' and second_closest_object['object_type'] == 'red_buoy') or \
                (closest_object['object_type'] == 'red_buoy' and second_closest_object['object_type'] == 'green_buoy'):
            rospy.logdebug("found gate")
            return True
        else:
            rospy.logdebug("no gate found")
            return False

    def check_gate_orientation(self):
        rospy.logdebug("Function called: check_gate_orientation")

        # Fetching the two closest objects
        closest_object = self.get_nth_object(1)
        second_closest_object = self.get_nth_object(2)

        # Check if both buoys are detected
        if not closest_object or not second_closest_object:
            rospy.logerr("Can't check gate orientation without both buoys.")
            return False

        # Check for expected object types
        expected_types = {'red_buoy', 'green_buoy'}
        if closest_object['object_type'] not in expected_types or second_closest_object['object_type'] not in expected_types:
            rospy.logerr("Unexpected object types. Expecting 'red_buoy' and 'green_buoy'.")
            return False

        # Determine buoy positions based on bearing
        is_red_on_left = closest_object['object_type'] == 'red_buoy' and (
                    closest_object['bearing'] - second_closest_object['bearing']) < 0

        is_green_on_left = closest_object['object_type'] == 'green_buoy' and (
                    closest_object['bearing'] - second_closest_object['bearing']) > 0

        if is_red_on_left or is_green_on_left:
            rospy.logdebug("Correct orientation: Red buoy is on the left, and green buoy is on the right.")
            return True
        else:
            rospy.logdebug("Incorrect orientation: Green buoy is on the left, and red buoy is on the right.")
            return False

    def relative_distance_within_threshold(self, n1, n2, relative_distance_threshold):

        try:
            first_object = self.get_nth_object(n1)
            second_object = self.get_nth_object(n2)

            if not first_object or not second_object:
                rospy.logdebug("Insufficient data to calculate relative distance.")
                return None

            a = first_object['distance']
            c = second_object['distance']
            beta = np.radians(first_object['bearing'] - second_object['bearing'])
            relative_distance = np.sqrt(a ** 2 + c ** 2 - 2 * a * c * np.cos(beta))

            rospy.logdebug(f"Relative distance between objects {n1} and {n2}: {relative_distance}")

            if relative_distance < relative_distance_threshold:
                rospy.logdebug("Relative distance within threshold.")
                return True
            else:
                rospy.logdebug("Relative distance exceeds threshold")
                return False
        except TypeError as e:
            rospy.logerr(f"Invalid value for distance or bearing: {e}")
            return None

    def nth_object_GPS_coordinates(self, n=1, earth_radius=6378100):

        nth_object = self.get_nth_object(n)

        if nth_object:
            nth_object_gps = DroneControl.convert_distance_to_gps_coordinates(
                nth_object['distance'], nth_object['bearing'],
                self.communication.lat, self.communication.lon, earth_radius)
            rospy.logdebug(f"Closest buoy GPS: {nth_object_gps}")
            return nth_object_gps
        else:
            rospy.logerr(f"{n}. closest object not detected!")
            return None
        
    @staticmethod
    def convert_distance_to_gps_coordinates(distance, bearing_degrees, latitude, longitude, earth_radius=6371000):

        latitude_rad = np.radians(latitude)
        longitude_rad = np.radians(longitude)
        bearing_rad = np.radians(bearing_degrees)

        # Calculate new latitude in radians
        shifted_latitude_rad = np.arcsin(np.sin(latitude_rad) * np.cos(distance / earth_radius) + np.cos(latitude_rad) * np.sin(distance / earth_radius) * np.cos(bearing_rad))

        # Calculate new longitude in radians
        shifted_longitude_rad = longitude_rad + np.arctan2(np.sin(bearing_rad) * np.sin(distance / earth_radius) * np.cos(latitude_rad), np.cos(distance / earth_radius) - np.sin(latitude_rad) * np.sin(shifted_latitude_rad))

        # Convert new latitude and longitude from radians to degrees
        shifted_latitude_degree = np.degrees(shifted_latitude_rad)
        shifted_longitude_degree = np.degrees(shifted_longitude_rad)

        # Round the degrees
        return round(shifted_latitude_degree, 6), round(shifted_longitude_degree, 6)

    @staticmethod
    def get_two_point_bearing(lat1, lon1, lat2, lon2):

        # Convert latitude and longitude from degrees to radians
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

        delta_lon_rad = lon2_rad - lon1_rad

        x = np.sin(delta_lon_rad) * np.cos(lat2_rad)
        y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(delta_lon_rad)

        bearing = np.degrees(np.arctan2(x, y)) % 360

        rospy.logdebug(f"Initial bearing between points: {bearing} degrees")
        return bearing

    @staticmethod
    def adjust_bearing_to_drone_heading(bearing, relative_angle, drone_heading):

        # Validate input ranges
        if not (0 <= bearing < 360) or not (0 <= drone_heading < 360):
            rospy.logerr("Invalid input: Bearing and drone heading must be within [0, 360).")
            return None

        drone_heading = drone_heading % 360
        adjusted_heading = (bearing + relative_angle - drone_heading + 360) % 360

        return adjusted_heading

    def reset_search_variables(self):
        self.steerboard_clear = False
        self.port_clear = False
        self.search_iter = 0

    def im_lost(self):
        if self.search_iter >= self.max_searches:
            return True
        else:
            return False

    def start_timer(self, sec):
        self.start_time = datetime.now() + timedelta(seconds=sec)

    def timer_reached(self):
        if datetime.now() >= self.start_time:
            return True
        else:
            return False

    def set_closest_gate_waypoints(self):
        """This function is only called when a gate has been confirmed. It calculates """
        first_object_coordinates = self.nth_object_GPS_coordinates(1)
        second_object_coordinates = self.nth_object_GPS_coordinates(2)

        self.waypoint_latitude = round((first_object_coordinates[0] + self.second_object_coordinates[0]) / 2, self.GPS_round)
        self.waypoint_longitude = round((first_object_coordinates[1] + self.second_object_coordinates[1]) / 2, self.GPS_round)

        # Bearing between the two buoys.
        self.buoys_bearing = self.get_two_point_bearing(first_object_coordinates[0], first_object_coordinates[1],
                                                        second_object_coordinates[0], second_object_coordinates[1])
        rospy.logdebug(f"Buoys bearing: {self.buoys_bearing}")

        # Calculate the heading to exit the gate.
        out_gate_heading = self.adjust_bearing_to_drone_heading(self.buoys_bearing, 90, self.communication.heading)
        rospy.logdebug(f"Out of gate heading: {out_gate_heading}")

        # Calculate the GPS coordinates to exit the gate.
        self.waypoint_latitude_out, self.waypoint_longitude_out = self.convert_distance_to_gps_coordinates(self.gate_exit_clearance, out_gate_heading, self.waypoint_latitude, self.waypoint_longitude)

        # Calculate the return point by mirroring the exit point across the midpoint.
        self.waypoint_latitude_out_return = 2 * self.waypoint_latitude - self.waypoint_latitude_out
        self.waypoint_longitude_out_return = 2 * self.waypoint_longitude - self.waypoint_longitude_out

        rospy.loginfo(f"Waypoint: ({self.waypoint_latitude}, {self.waypoint_longitude}), exit waypoint: ({self.waypoint_latitude_out}, {self.waypoint_longitude_out}), return waypoint: ({self.waypoint_latitude_out_return}, {self.waypoint_longitude_out_return})")

    def obstacle_channel_yellow_buoy(self):
        # Bearing from drone to the closest buoy.
        drone_buoy_bearing = self.get_two_point_bearing(self.communication.lat, self.communication.lon, self.object_GPS[0], self.object_GPS[1])

        # Calculate waypoint to navigate around the yellow buoy.
        dist_to_wp = np.sqrt(self.closest_dist ** 2 + self.speed_gate_waypoint_clearance ** 2)
        angle_buoy_wp = np.degrees(np.arctan2(self.speed_gate_waypoint_clearance * np.sign(self.closest_bearing), self.closest_dist))
        self.wp_yellow_buoy_lat, self.wp_yellow_buoy_lon = self.convert_distance_to_gps_coordinates(dist_to_wp, drone_buoy_bearing + angle_buoy_wp, self.communication.lat, self.communication.lon)

        rospy.loginfo(f"Navigating around yellow_buoy, waypoint set at: ({self.wp_yellow_buoy_lat}, {self.wp_yellow_buoy_lon})")

    def speed_gate_yellow_buoy(self):
        # Bearing from drone to the closest buoy.
        drone_buoy_bearing = self.get_two_point_bearing(self.communication.lat, self.communication.lon, self.object_GPS[0], self.object_GPS[1])

        # Side decision based on closest buoy bearing.
        side_multiplier = -1 if self.closest_bearing < 0 else 1

        # Calculate waypoints for the speed gate challenge.
        for i, dist_multiplier in enumerate([(1, 1), (1, 0), (1, -1)], start=1):
            dist_to_wp = np.sqrt(self.closest_dist ** 2 + (self.speed_gate_waypoint_clearance * dist_multiplier[1]) ** 2)
            angle_buoy_wp = side_multiplier * np.degrees(np.arctan2(self.speed_gate_waypoint_clearance * dist_multiplier[1], self.closest_dist))
            setattr(self, f"yellow_buoy_lat_{i}", self.convert_distance_to_gps_coordinates(dist_to_wp, drone_buoy_bearing + angle_buoy_wp, self.communication.lat, self.communication.lon)[0])
            setattr(self, f"yellow_buoy_lon_{i}", self.convert_distance_to_gps_coordinates(dist_to_wp, drone_buoy_bearing + angle_buoy_wp, self.communication.lat, self.communication.lon)[1])

        rospy.loginfo(f"Waypoints for speed gate challenge set around the yellow buoy.")

    def detect_and_traverse_gates(self):
        """Looks for gates and sets waypoints. If no waypoints have been set, returns false."""
        if self.check_buoy_gate():
            if self.check_gate_orientation():
                self.handle_gate_passage()
            else:
                rospy.loginfo("Gate detected but wrong orientation, rotating 180 degrees.")
                self.rotate_drone(180)
                return False
        elif self.closest_color in ["red_buoy", "green_buoy"] and self.second_is_none:
            rospy.logwarn("Single buoy detected, approaching for better identification.")
            self.approach_buoy(self.object_GPS)
        else:
            return False  # Indicate that no buoy-related action was taken
        return True

    def handle_gate_passage(self):
        self.set_closest_gate_waypoints()
        self.set_navigation_waypoints([self.waypoint_latitude, self.waypoint_longitude], [self.waypoint_latitude_out, self.waypoint_longitude_out], "AUTO")
        self.start_timer_based_on_distance(self.closest_dist, additional_time=7)

    def handle_yellow_buoy(self):
        if self.second_is_none or self.relative_distance_within_threshold():
            rospy.loginfo("Navigating around yellow buoy.")
            self.obstacle_channel_yellow_buoy()
            self.send_guided_waypoint(self.wp_yellow_buoy_lat, self.wp_yellow_buoy_lon)
            self.start_timer_based_on_distance(self.closest_dist, additional_time=5)
        else:
            rospy.loginfo("Yellow buoy detected, but close to another buoy, approaching cautiously.")
            self.approach_buoy(self.object_GPS, reduce_speed=True)

    def handle_no_detections(self):
        if not self.steerboard_clear:
            rospy.loginfo("No detections, turning starboard.")
            self.rotate_drone(self.look_degrees)
            self.steerboard_clear = True
        elif not self.port_clear:
            rospy.loginfo("No detections, turning port.")
            self.rotate_drone(2*(-self.look_degrees))
            self.port_clear = True
        elif self.im_lost():
            rospy.logerr("Lost, attempting to return to last known gate exit.")
            if self.waypoint_latitude is not None and self.waypoint_latitude_out is not None:
                # Go back to last gate if no detections are made
                self.set_navigation_waypoints([self.waypoint_latitude, self.waypoint_longitude],
                                              [self.waypoint_latitude_out, self.waypoint_longitude_out], "AUTO")
        else:
            rospy.logwarn("No detections, moving forward to scan again.")
            self.rotate_drone(self.look_degrees)
            self.move_forward_scan()

    def reset_waypoints(self):
        """Initiate a new scan for buoys or gates."""
        self.communication.clear_guided_wp()
        self.communication.clear_waypoints()
        self.communication.wp_set = False

    def rotate_drone(self, degrees):
        """Rotate the drone a certain number of degrees."""
        self.communication.change_mode("GUIDED")
        self.communication.rotate_x_deg(degrees, 60)
        self.reset_search_variables()

    def approach_buoy(self, gps_coords, reduce_speed=False):
        """Approach a buoy to get a better look or navigate around it."""
        self.communication.change_mode("GUIDED")
        speed_factor = 0.5 if reduce_speed else 1
        self.communication.send_guided_wp(gps_coords[0], gps_coords[1], speed_factor)
        self.start_timer((self.closest_dist / self.communication.max_speed) * speed_factor)
        self.reset_search_variables()

    def set_navigation_waypoints(self, initial_wp, final_wp, mode="GUIDED"):
        """Set a sequence of waypoints for navigation."""
        self.communication.change_mode(mode)
        self.communication.clear_waypoints()
        self.communication.make_waypoint(*initial_wp, curr=True)
        self.communication.make_waypoint(*final_wp)
        self.communication.send_waypoint()
        self.communication.change_mode("AUTO")

    def start_timer_based_on_distance(self, distance, additional_time=0):
        """Start a timer based on distance to travel and an optional additional time."""
        self.start_timer(distance / self.communication.max_speed + additional_time)
        self.reset_search_variables()

    def move_forward_scan(self):
        """Move forward to continue scanning for buoys or gates."""
        self.waypoint_latitude, self.waypoint_longitude = self.convert_distance_to_gps_coordinates(self.search_speed, self.communication.heading,
                                                                                                   self.communication.lat, self.communication.lon)
        self.send_guided_waypoint(self.waypoint_latitude, self.waypoint_longitude)
        self.steerboard_clear = False
        self.port_clear = False
        self.search_iter += 1

    def send_guided_waypoint(self, lat, lon):
        """Send a single guided waypoint."""
        self.communication.change_mode("GUIDED")
        self.communication.send_guided_wp(lat, lon)

    def setup_speed_gate_course(self):
        self.speed_gate_yellow_buoy()
        self.set_guided_waypoints([
            (self.yellow_buoy_lat_1, self.yellow_buoy_lon_1),
            (self.yellow_buoy_lat_2, self.yellow_buoy_lon_2),
            (self.yellow_buoy_lat_3, self.yellow_buoy_lon_3),
            (self.start_lat, self.start_lon),
            (self.start_lat_out_ret, self.start_lon_out_ret)
        ], mode="AUTO")
        self.yellow_set = True

    def setup_initial_gate_passage(self):
        self.set_closest_gate_waypoints()
        self.set_guided_waypoints([
            (self.waypoint_latitude, self.waypoint_longitude),
            (self.waypoint_latitude_out, self.waypoint_longitude_out)
        ], mode="AUTO")
        # Store starting position for return journey
        self.start_lat, self.start_lon = self.waypoint_latitude, self.waypoint_longitude
        self.start_lat_out_ret, self.start_lon_out_ret = self.waypoint_latitude_out_return, self.waypoint_longitude_out_return
        self.gate_set = True

    def advance_towards_next_detection(self):
        # If no yellow buoy detected, move towards the next point for detection.
        self.waypoint_latitude, self.waypoint_longitude = self.convert_distance_to_gps_coordinates(self.sg_no_buoy_dist, self.communication.heading, self.communication.lat, self.communication.lon)
        self.send_guided_waypoint(self.waypoint_latitude, self.waypoint_longitude)

    def set_guided_waypoints(self, waypoints, mode="GUIDED"):
        self.communication.change_mode(mode)
        self.communication.clear_waypoints()
        for lat, lon in waypoints:
            self.communication.make_waypoint(lat, lon, curr=waypoints.index((lat, lon)) == 0)
        self.communication.send_waypoint()
        if mode == "AUTO":
            self.communication.change_mode(mode)

    def complete_mission(self):
        rospy.loginfo("Mission complete, returning home.")
        self.communication.RTL()

    def initiate_docking_procedure(self):
        try:
            if self.check_buoy_gate():
                self.set_closest_gate_waypoints()
                self.send_guided_waypoint(self.waypoint_latitude_out_return, self.waypoint_longitude_out_return)
                self.docking_hdg = self.buoys_bearing
        except IndexError:
            pass

    def start_undocking_sequence(self):
        self.start_timer(5)  # Adjust the time as needed for undocking duration
        self.docking_timer = False
        self.second_timer = True

    def needs_heading_correction(self):
        return (self.docking_hdg - 5 > self.communication.heading or
                self.docking_hdg + 5 < self.communication.heading)

    def correct_heading_towards_docking(self):
        correction_direction = 5 if self.docking_hdg - 5 > self.communication.heading else -5
        rospy.loginfo("Correcting heading...")
        self.communication.rotate_x_deg(self.docking_hdg, correction_direction)

    def reset_docking_state(self):
        self.start_docking = False
        self.docking_timer = False
        self.second_timer = False

# --------------------------------------------- Autopilot Communication --------------------------------------------- #


class AutoPilotCommunication:
    heading = None
    lon = None
    lat = None
    wp_set = None
    max_speed = 1.6  # Vessel max speed in m/s
    NAV_COMMAND = 16  # Navigation command
    RTL_COMMAND = 20  # Return to Launch command
    GLOBAL_FRAME = 0  # Global frame identifier

    def __init__(self):
        
        # Publishers
        self.pub_vel = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel_unstamped", Twist, queue_size=10)
        self.pub_guided_wp = rospy.Publisher("/mavros/setpoint_raw/global", GlobalPositionTarget, queue_size=10)

        # Subscribers
        self.sub_state = rospy.Subscriber("/mavros/state", State, self.state_callback)
        self.sub_GPS = rospy.Subscriber("/mavros/global_position/global", NavSatFix, self.gps_callback)
        self.sub_heading = rospy.Subscriber("/mavros/global_position/compass_hdg", Float64, self.heading_callback)
        self.sub_wp_reached = rospy.Subscriber("/mavros/mission/reached", WaypointReached, self.wp_reached_callback)
        self.sub_wps = rospy.Subscriber("/mavros/mission/waypoints", WaypointList, self.wps_callback)

        # Attributes
        self.cmd_vel = Twist()
        self.guided_wp = GlobalPositionTarget()
        self.is_connected = False
        self.is_armed = False
        self.ini_mode = None
        self.ctrl_c = False
        self.arming = False
        self.mode = 'MANUAL'
        self.lat = None
        self.lon = None
        self.wp_list = []
        self.wps = None
        self.heading = None
        self.wp_reached = False
        self.reached_seq = None
        self.curr_seq = None
        self.wp_set = False

        rospy.on_shutdown(self.shutdownhook)
        
        print("wait for connection")
        self.wait_for_connection()
        print("connected")
        # Logging initial statuses
        rospy.loginfo(f"MAVROS connection status: {self.is_connected}")
        rospy.loginfo(f"MAVROS armed status: {self.is_armed}")
        rospy.loginfo(f"MAVROS initial mode: {self.ini_mode}")

    def wait_for_connection(self, timeout=30):
        start_time = rospy.get_time()
        while not rospy.is_shutdown() and not self.is_connected and (rospy.get_time() - start_time) < timeout:
            rospy.loginfo("Waiting for MAVROS connection...")
            rospy.sleep(1)
        if not self.is_connected:
            rospy.logwarn("Failed to connect to MAVROS within the timeout period.")
    def shutdownhook(self):
        self.ctrl_c = True

    def state_callback(self, data):
        self.is_connected = data.connected
        self.is_armed = data.armed
        self.ini_mode = data.mode

    def arm(self, status):
        """ Arms or disarms the vehicle based on the status. """
        rospy.wait_for_service('/mavros/cmd/arming')
        try:
            if self.arming != status:
                arming_client = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
                arming_obj = CommandBoolRequest()
                arming_obj.value = status
                response = arming_client(arming_obj)
                if response.success:  # Assuming the CommandBool service returns an object with a 'success' attribute
                    rospy.loginfo(f"Arming status successfully changed to {status}.")
                else:
                    rospy.logwarn(f"Arming command was sent but not successful. Response: {response.message}")
                self.arming = status
            else:
                rospy.logwarn("Vessel is already armed with the desired status!")
        except rospy.ServiceException as arm_exception:
            rospy.logerr(f"Arming failed: {arm_exception}")

    def change_mode(self, new_mode):
        """ Changes the operational mode of the vehicle. """
        rospy.wait_for_service('/mavros/set_mode')
        try:
            if self.mode != new_mode:
                change_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)
                mode_obj = SetModeRequest()
                mode_obj.custom_mode = new_mode
                response = change_mode_client(mode_obj)
                if response.mode_sent:  # Assuming the SetMode service returns an object with a 'mode_sent' attribute
                    rospy.loginfo(f"Mode changed to {new_mode} successfully.")
                else:
                    rospy.logwarn(
                        f"Mode change to {new_mode} command sent but not successful. Response: {response.message}")
                self.mode = new_mode
            else:
                rospy.logwarn("Vessel is already in the desired mode!")
        except rospy.ServiceException as mode_exception:
            rospy.logerr(f"Mode change failed: {mode_exception}")

    def gps_callback(self, msg):
        """Updates the current GPS coordinates of the drone and logs them."""
        self.lat = msg.latitude
        self.lon = msg.longitude
        rospy.logdebug(f"Drone GPS location: {self.lat}, {self.lon}")

    def heading_callback(self, msg):
        """Updates the current heading of the drone and logs    it."""
        self.heading = msg.data
        rospy.logdebug(f"Drone heading: {self.heading}")

    def wps_callback(self, msg):
        """Updates the current sequence and waypoint list of the drone and logs the waypoints."""
        self.curr_seq = msg.current_seq
        self.wps = msg.waypoints
        rospy.logdebug(f"Waypoint list: {self.wps}")

    def wp_reached_callback(self, msg):
        """Updates the drone's waypoint reaching status and logs the sequence of the reached waypoint."""
        self.reached_seq = msg.wp_seq
        self.wp_reached = True
        rospy.loginfo("Waypoint reached!")
        rospy.logdebug(f"Waypoint reached: Sequence {self.reached_seq}")

    def auto_wp_reached(self):
        """ Checks if the current waypoint in AUTO mode has been reached. """
        if self.mode == "AUTO" and self.curr_seq == self.reached_seq:
            rospy.loginfo("AUTO waypoint reached!")
            return True
        rospy.logdebug(
            f"Not in AUTO mode or current sequence {self.curr_seq} does not match reached sequence {self.reached_seq}.")
        return False

    def guided_wp_reached(self):
        """ Checks if the current waypoint in GUIDED mode has been reached. """
        if self.wp_reached and self.mode == "GUIDED":
            rospy.loginfo("GUIDED waypoint reached!")
            return True
        rospy.logdebug("GUIDED waypoint not reached or not in GUIDED mode.")
        return False

    def waypoint_reached(self):
        """ Determines if any waypoint has been reached, resets waypoint reached state if true. """
        auto_reached = self.auto_wp_reached()
        guided_reached = self.guided_wp_reached()
        if auto_reached or guided_reached:
            rospy.loginfo("Waypoint reached, resetting waypoint reach status.")
            self.wp_reached = False
            return True
        rospy.logdebug("No waypoint reached.")
        return False

    def make_waypoint(self, lat, lon, cmd=NAV_COMMAND, is_current=False, auto_continue=True):
        """ Creates a waypoint and appends it to the waypoint list. """
        wp = Waypoint()
        wp.frame = self.GLOBAL_FRAME  # Using the class constant for frame type
        wp.command = cmd
        wp.is_current = is_current
        wp.autocontinue = auto_continue
        wp.param1 = 0  # HOLD time at waypoint
        wp.param2 = 0  # Acceptance radius, if inside waypoint count as reached
        wp.param3 = 0  # Pass Radius. If 0, go through the waypoint
        wp.param4 = 0  # Yaw angle, 0 for our use case (e.g., USV - Unmanned Surface Vehicle)
        wp.x_lat = lat
        wp.y_long = lon
        wp.z_alt = 0  # Altitude, not used in this context

        self.wp_list.append(wp)
        rospy.loginfo(f"Waypoint created at ({lat}, {lon}) with command {cmd}.")

    def send_waypoint(self):
        """Sends the list of waypoints to the MAVROS mission push service."""
        rospy.wait_for_service('mavros/mission/push')
        try:
            service_request = rospy.ServiceProxy('mavros/mission/push', WaypointPush)
            service_response = service_request(start_index=0, waypoints=self.wp_list)
            self.wp_set = service_response.success
            if self.wp_set:
                rospy.loginfo("Waypoint successfully pushed to MAVROS.")
            else:
                rospy.logerr(f"FAILURE: PUSHING WP! Server response: {service_response.message}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to send WayPoint: {e}")

    def clear_waypoints(self):
        """Clears all waypoints from the MAVROS mission and resets the local waypoint list."""
        rospy.wait_for_service('mavros/mission/clear')
        try:
            clear_service = rospy.ServiceProxy('mavros/mission/clear', WaypointClear)
            response = clear_service.call()
            if response.success:
                rospy.loginfo("Waypoints successfully cleared from MAVROS.")
                self.wp_list = []  # Clear the local list only after confirming the operation was successful
                self.make_waypoint(0, 0)  # Add dummy waypoint to list, if required for initialization purposes
                return True
            else:
                rospy.logerr(f"Waypoint clearing failed with server response: {response.message}")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"Clear waypoint failed: {e}")
            return False

    def send_guided_wp(self, lat, lon):
        """Publishes a guided waypoint and ensures it is sent by checking subscriber connections."""
        self.guided_wp.latitude = lat
        self.guided_wp.longitude = lon
        while not self.ctrl_c:
            connections = self.pub_guided_wp.get_num_connections()
            if connections > 0:
                self.pub_guided_wp.publish(self.guided_wp)
                rospy.loginfo(f"GUIDED waypoint published at ({lat}, {lon}).")
                self.wp_set = True
                break
            else:
                rospy.logdebug("No subscribers to guided_wp yet, waiting to try again...")
                rospy.sleep(0.2)  # Sleep to avoid busy waiting and reduce CPU load

    def clear_guided_wp(self):
        """Clears any guided waypoint by toggling between MANUAL and GUIDED modes."""
        if not self.change_mode("MANUAL"):
            rospy.logerr("Failed to change to MANUAL mode in clear_guided_wp.")
        if not self.change_mode("GUIDED"):
            rospy.logerr("Failed to change back to GUIDED mode in clear_guided_wp.")

    def RTL(self):
        """
        Executes a Return-To-Launch procedure """
        if not self.change_mode("GUIDED"):
            rospy.logerr("Failed to switch to GUIDED mode for RTL.")
            return False

        if not self.clear_waypoints():
            rospy.logerr("Failed to clear waypoints for RTL.")
            return False

        self.make_waypoint(0, 0, cmd=20)  # Assuming 0, 0 is the home position
        if not self.send_waypoint():
            rospy.logerr("Failed to send home waypoint for RTL.")
            return False

        if not self.change_mode("AUTO"):
            rospy.logerr("Failed to switch to AUTO mode after setting home waypoint for RTL.")
            return False

        rospy.loginfo("RTL procedure successfully initiated.")
        return True

    def rotate_x_deg(self, x_deg, rate):
        """Rotates the drone a specific number of degrees relative to its current heading."""
        tolerance = round(0.25 * abs(rate), 2)
        target_heading_low = (self.heading + x_deg - tolerance) % 360
        target_heading_high = (self.heading + x_deg + tolerance) % 360

        while not self.ctrl_c:
            current_heading = self.heading
            if target_heading_low < target_heading_high:
                if target_heading_low <= current_heading <= target_heading_high:
                    break
            else:
                if current_heading >= target_heading_low or current_heading <= target_heading_high:
                    break

            self.rotate(rate)
            rospy.sleep(0.1)
            rospy.logdebug(
                f"Current heading: {current_heading}, Target range: [{target_heading_low}, {target_heading_high}]")

        self.stop()
        rospy.loginfo(f"Rotation to {x_deg} degrees at rate {rate} completed.")

    def rotate(self, deg_s):
        """ Rotates the drone around the z-axis at a specified rate in degrees per second.
        Positive values cause starboard rotation"""
        if not -360 <= deg_s <= 360:
            rospy.logerr(
                f"Requested rotation rate {deg_s} out of bounds. Must be between -360 and 360 degrees per second.")
            return

        # Reset velocities to ensure no unintended movement
        self.cmd_vel.linear.x = 0
        self.cmd_vel.linear.y = 0
        self.cmd_vel.linear.z = 0
        self.cmd_vel.angular.x = 0
        self.cmd_vel.angular.y = 0
        self.cmd_vel.angular.z = np.radians(deg_s)  # Convert degrees to radians for ROS

        self.pub_vel.publish(self.cmd_vel)
        rospy.loginfo(f"Initiating rotation at {deg_s} degrees per second.")

    def move_forward(self, lin_vel):
        """ Moves the drone forward along the Y-axis at a specified linear velocity. """
        # Reset all other velocities to zero to prevent unintended motion
        self.cmd_vel.linear.x = 0
        self.cmd_vel.linear.z = 0
        self.cmd_vel.angular.x = 0
        self.cmd_vel.angular.y = 0
        self.cmd_vel.angular.z = 0

        # Set the forward velocity
        self.cmd_vel.linear.y = lin_vel
        self.pub_vel.publish(self.cmd_vel)
        rospy.loginfo(f"Drone moving forward at {lin_vel} m/s.")

    def move_sideways(self, lin_vel):
        """ Moves the drone sideways along the X-axis at a specified linear velocity. """
        # Reset all other velocities to zero to ensure no unintended motion
        self.cmd_vel.linear.y = 0
        self.cmd_vel.linear.z = 0
        self.cmd_vel.angular.x = 0
        self.cmd_vel.angular.y = 0
        self.cmd_vel.angular.z = 0

        # Set the sideways velocity
        self.cmd_vel.linear.x = lin_vel
        self.pub_vel.publish(self.cmd_vel)
        rospy.loginfo(f"Drone moving sideways at {lin_vel} m/s.")

    def stop(self):
        """ Stops all drone movements by setting all velocity components to zero and ensuring the stop command is published."""
        # Reset all velocity components
        self.cmd_vel.linear.x = 0
        self.cmd_vel.linear.y = 0
        self.cmd_vel.linear.z = 0
        self.cmd_vel.angular.x = 0
        self.cmd_vel.angular.y = 0
        self.cmd_vel.angular.z = 0

        # Attempt to publish the stop command, with retries and sleep to avoid busy waiting
        retry_count = 0
        max_retries = 50
        while not self.ctrl_c and retry_count < max_retries:
            connections = self.pub_vel.get_num_connections()
            if connections > 0:
                self.pub_vel.publish(self.cmd_vel)
                rospy.loginfo("Motors stopped.")
                return True
            else:
                rospy.sleep(0.2)  # Sleep to reduce CPU load and wait for subscribers to connect
                rospy.logdebug("No subscribers to pub_vel yet, retrying...")
                retry_count += 1

        rospy.logerr("Failed to stop motors: No subscribers to pub_vel after retries.")
        return False