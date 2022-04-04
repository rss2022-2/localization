#!/usr/bin/env python2

import rospy
from sensor_model import SensorModel
from motion_model import MotionModel

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseArray, Pose
from localization.msg import PoseError
import tf2_ros

import numpy as np

from threading import Semaphore

class ParticleFilter:

    def __init__(self):
        # Get parameters
        self.particle_filter_frame = rospy.get_param("~particle_filter_frame")
        self.num_particles = rospy.get_param("~num_particles", 200)
        self.lidar_reduce_factor = rospy.get_param("~lidar_reduce_factor", 5)
        self.num_beams = rospy.get_param("~num_beams_per_particle", 100)
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
        self.use_thread = rospy.get_param("~use_thread", True)
        if self.use_thread:
            rospy.loginfo("Use thread")

        self.semaphore = Semaphore()
        self.count = 0

        # Initialize the models
        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.
        self.started = True
        #self.init_particles = np.zeros((self.num_particles, 3))
        self.particles = np.full((self.num_particles,3), [-13.144, 13.835, 0.823])
        self.probabilities = np.ones(self.num_particles)/self.num_particles
        self.norm_probabilities = np.ones(self.num_particles)/self.num_particles
        self.odom_msg = Odometry()
        self.odom_msg.header.seq = 0
        self.odom_msg.header.frame_id = "/map"
        self.odom_msg.child_frame_id = self.particle_filter_frame
        
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.transform_msg = TransformStamped()
        self.transform_msg.header.frame_id = "/map"
        self.transform_msg.child_frame_id = self.particle_filter_frame

        # ground truth pose for error plot
        self.ground_odom_pose = None

        self.laser_sub = rospy.Subscriber(scan_topic, LaserScan,
                                          self.lidar_callback, # TODO: Fill this in
                                          queue_size=1)
        self.odom_sub  = rospy.Subscriber(odom_topic, Odometry,
                                          self.odom_callback, # TODO: Fill this in
                                          queue_size=1)
        self.odom_time = rospy.Time.now()

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

        self.PoseArray_pub = rospy.Publisher("/pose_array", PoseArray, queue_size = 1)

        self.error_pub = rospy.Publisher("/pose_error", PoseError, queue_size=10)


    def lidar_callback(self, lidar_msg):
        if self.count % self.lidar_reduce_factor == 0 and self.started:
            if not self.use_thread or self.semaphore.acquire():
                observation = np.array(lidar_msg.ranges)
                down_sampled_observation = observation[np.linspace(0, len(observation) - 1, self.num_beams, dtype=int)]
                probabilities = self.sensor_model.evaluate(self.particles, down_sampled_observation)
                if probabilities is not None:
                    self.norm_probabilities = probabilities/sum(probabilities)
                    selected_indices = np.random.choice(self.num_particles, self.num_particles, p=self.norm_probabilities)
                    self.particles = self.particles[selected_indices]
                    self.probabilities = probabilities[selected_indices]
                    self.particles = self.motion_model.add_noise(self.particles, [0.15, 0.15, 0.3])
                    self.pose_odom_update()
                if self.use_thread:
                    self.semaphore.release()
        self.count = (1 + self.count) % self.lidar_reduce_factor


    def odom_callback(self, odom_msg):
        if self.started:
            time_diff = odom_msg.header.stamp - self.odom_time
            dx = odom_msg.twist.twist.linear.x * time_diff.to_sec()
            dy = odom_msg.twist.twist.linear.y * time_diff.to_sec()
            dt = odom_msg.twist.twist.angular.z * time_diff.to_sec()
            self.odom_time = odom_msg.header.stamp
            if not self.use_thread or self.semaphore.acquire():
                self.particles = np.array(self.motion_model.evaluate(self.particles, [dx, dy, dt]))
                self.pose_odom_update()
                if self.use_thread:
                    self.semaphore.release()
            self.ground_odom_pose = odom_msg.pose

    def initpose_callback(self, pose_msg):
        dx = pose_msg.pose.pose.position.x
        dy = pose_msg.pose.pose.position.y
        dt = euler_from_quaternion([
            pose_msg.pose.pose.orientation.x,
            pose_msg.pose.pose.orientation.y,
            pose_msg.pose.pose.orientation.z,
            pose_msg.pose.pose.orientation.w
        ])[2]
        self.semaphore.acquire()
        self.particles = np.full((self.num_particles, 3), [dx, dy, dt])
        self.started = True
        self.semaphore.release()


    def pose_odom_update(self):
        average_pose = self.__get_average_pose()
        [qx, qy, qz, qw] = quaternion_from_euler(0, 0, average_pose[2])

        self.transform_msg.transform.translation.x = average_pose[0]
        self.transform_msg.transform.translation.y = average_pose[1]
        self.transform_msg.transform.translation.z = 0
        self.transform_msg.transform.rotation.x = qx
        self.transform_msg.transform.rotation.y = qy
        self.transform_msg.transform.rotation.z = qz
        self.transform_msg.transform.rotation.w = qw
        self.transform_msg.header.stamp = rospy.Time.now()
        self.tf_broadcaster.sendTransform(self.transform_msg)

        self.odom_msg.pose.pose.position.x = average_pose[0]
        self.odom_msg.pose.pose.position.y = average_pose[1]
        self.odom_msg.pose.pose.orientation.x = qx
        self.odom_msg.pose.pose.orientation.y = qy
        self.odom_msg.pose.pose.orientation.z = qz
        self.odom_msg.pose.pose.orientation.w = qw

        # self.odom_msg.header.seq += 1
        self.odom_msg.header.stamp = rospy.Time.now()
        self.odom_pub.publish(self.odom_msg)
        
        PoseArray_msg = PoseArray()
        PoseArray_msg.header.stamp = rospy.Time.now()
        PoseArray_msg.header.frame_id = "/map"
        PoseArray_msg.poses = ParticleFilter.__particles_to_poses(self.particles)
        self.PoseArray_pub.publish(PoseArray_msg)
        # self.error_publisher(self.odom_msg)


    def __get_average_pose(self):
        # self.semaphore.acquire()
        #average_t = ParticleFilter.__get_average_angle(self.particles[:,2], self.norm_probabilities)
        #average_x = np.dot(self.particles[:,0], self.norm_probabilities) - 0.275*np.cos(average_t)
        #average_y = np.dot(self.particles[:,1], self.norm_probabilities) - 0.275*np.sin(average_t)
        average_t = ParticleFilter.__get_average_angle(self.particles[:,2], self.norm_probabilities)
        average_x = np.average(self.particles[:,0]) #- 0.275*np.cos(average_t)
        average_y = np.average(self.particles[:,1]) #- 0.275*np.sin(average_t)
        # self.semaphore.release()

        return [average_x, average_y, average_t]
    
    def error_publisher(self, estimated_odom_msg):
        error_msg = PoseError()
        if (self.ground_odom_pose is None):
            error_msg.distance_err = 0
            error_msg.rotation_err = 0
            self.error_pub.publish(error_msg)
            return

        error_msg = PoseError()
        estimated_x = estimated_odom_msg.pose.pose.position.x
        estimated_y = estimated_odom_msg.pose.pose.position.y

        ground_x = self.ground_odom_pose.pose.position.x
        ground_y = self.ground_odom_pose.pose.position.y

        distance = np.sqrt(np.square(estimated_x - ground_x) + np.square(estimated_y - ground_y))

        estimated_t = euler_from_quaternion([
            estimated_odom_msg.pose.pose.orientation.x,
            estimated_odom_msg.pose.pose.orientation.y,
            estimated_odom_msg.pose.pose.orientation.z,
            estimated_odom_msg.pose.pose.orientation.w
        ])[2]

        ground_t = euler_from_quaternion([
            self.ground_odom_pose.pose.orientation.x,
            self.ground_odom_pose.pose.orientation.y,
            self.ground_odom_pose.pose.orientation.z,
            self.ground_odom_pose.pose.orientation.w
        ])[2]

        error_msg.distance_err = distance
        error_msg.rotation_err = abs(ground_t - estimated_t)
        self.error_pub.publish(error_msg)

    @staticmethod    
    def __get_average_angle(angles, probabilities):
        #average_sin = np.dot(np.sin(angles), probabilities)
        #average_cos = np.dot(np.cos(angles), probabilities)
        average_sin = np.average(np.sin(angles))
        average_cos = np.average(np.cos(angles))
        return np.arctan2(average_sin, average_cos)

    @staticmethod
    def __particles_to_poses(particles):
        pose_array = []
        for p in particles:
            pose = Pose()
            pose.position.x = p[0]
            pose.position.y = p[1]
            pose.position.z = 0
            x, y, z, w = quaternion_from_euler(0, 0, p[2])
            pose.orientation.x = x
            pose.orientation.y = y
            pose.orientation.z = z
            pose.orientation.w = w
            pose_array.append(pose)

        return pose_array


if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
