#!/usr/bin/env python2
import rospy
from sensor_model import SensorModel
from motion_model import MotionModel

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_from_euler

import numpy as np

class ParticleFilter:
    NUM_PARTICLES = 50

    def __init__(self):
        # Get parameters
        self.particle_filter_frame = \
                rospy.get_param("~particle_filter_frame")

        # Initialize publishers/subscribers
        #
        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        scan_topic = rospy.get_param("~scan_topic", "/scan")
        odom_topic = rospy.get_param("~odom_topic", "/odom")

        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()
        self.particles = np.zeros((ParticleFilter.NUM_PARTICLES, 3))
        self.probabilities = np.ones(ParticleFilter.NUM_PARTICLES)/ParticleFilter.NUM_PARTICLES
        self.odom_msg = Odometry()
        self.odom_msg.header.seq = 0
        self.odom_msg.header.frame_id = "/map"
        self.odom_msg.child_frame_id = self.particle_filter_frame

        self.laser_sub = rospy.Subscriber(scan_topic, LaserScan,
                                          self.lidar_callback, # TODO: Fill this in
                                          queue_size=1)
        self.odom_sub  = rospy.Subscriber(odom_topic, Odometry,
                                          self.odom_callback, # TODO: Fill this in
                                          queue_size=1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.
        self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped,
                                          self.initpose_callback, # TODO: Fill this in
                                          queue_size=1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.
        self.odom_pub  = rospy.Publisher("/pf/pose/odom", Odometry, queue_size = 1)
        
        # Initialize the models
        # self.motion_model = MotionModel()
        # self.sensor_model = SensorModel()

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.
        # self.particles = np.zeros((ParticleFilter.NUM_PARTICLES, 3))
        # self.probabilities = np.ones(ParticleFilter.NUM_PARTICLES)/ParticleFilter.NUM_PARTICLES
        # self.odom_msg = Odometry()
        # self.odom_msg.header.seq = 0
        # self.odom_msg.header.frame_id = "/map"
        # self.odom_msg.child_frame_id = self.particle_filter_frame
        
        rospy.Timer(rospy.Duration(1.0/20.0), self.pose_odom_callback)


    def lidar_callback(self, lidar_msg):
        observation = lidar_msg.ranges
        probabilities = self.sensor_model.evaluate(self.particles, observation)
        normalized_probabilities = probabilities/sum(probabilities)
        selected_indices = np.random.choice(ParticleFilter.NUM_PARTICLES, ParticleFilter.NUM_PARTICLES, p=normalized_probabilities)
        self.particles = np.array([self.particles[i] for i in selected_indices])
        self.probabilities = np.array([probabilities[i] for i in selected_indices])


    def odom_callback(self, odom_msg):
        dx = odom_msg.twist.twist.linear.x
        dy = odom_msg.twist.twist.linear.y
        dt = odom_msg.twist.twist.angular.z
        self.particles = np.array(self.motion_model.evaluate(self.particles, [dx, dy, dt]))


    def initpose_callback(self, pose_msg):
        dx = pose_msg.pose.pose.position.x
        dy = pose_msg.pose.pose.position.x
        dt = euler_from_quaternion([
            pose_msg.pose.pose.orientation.x,
            pose_msg.pose.pose.orientation.y,
            pose_msg.pose.pose.orientation.z,
            pose_msg.pose.pose.orientation.w
        ])[0]
        self.particles = np.array(self.motion_model.evaluate(self.particles, [dx, dy, dt]))
        rospy.loginfo(self.particles)

    def pose_odom_callback(self, event):
        average_pose = self.__get_average_pose()
        
        self.odom_msg.pose.pose.position.x = average_pose[0]
        self.odom_msg.pose.pose.position.y = average_pose[1]
        [qx, qy, qz, qw] = quaternion_from_euler(0, 0, average_pose[2])
        self.odom_msg.pose.pose.orientation.x = qx
        self.odom_msg.pose.pose.orientation.y = qy
        self.odom_msg.pose.pose.orientation.z = qz
        self.odom_msg.pose.pose.orientation.w = qw

        self.odom_msg.header.seq += 1
        self.odom_msg.header.stamp = rospy.Time.now()
        self.odom_pub.publish(self.odom_msg)
        rospy.loginfo(self.odom_msg)


    def __get_average_pose(self):
        probabilities_square = self.probabilities**2
        average_probabilities = probabilities_square/sum(probabilities_square)
        weighted_xs = self.particles[:,0] * average_probabilities
        weighted_ys = self.particles[:,1] * average_probabilities
        weighted_ts = self.particles[:,2] * average_probabilities

        average_x = np.average(weighted_xs)
        average_y = np.average(weighted_ys)
        average_t = ParticleFilter.__get_average_angle(weighted_ts)

        return [average_x, average_y, average_t]
    
    
    @staticmethod    
    def __get_average_angle(angles):
        average_sin = np.average(np.sin(angles))
        average_cos = np.average(np.cos(angles))
        return np.arctan2(average_sin, average_cos)

if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
