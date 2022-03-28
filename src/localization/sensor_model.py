import numpy as np
from scan_simulator_2d import PyScanSimulator2D

import rospy
import tf
from nav_msgs.msg import OccupancyGrid
from tf.transformations import quaternion_from_euler

class SensorModel:


    def __init__(self):
        # Fetch parameters
        self.map_topic = rospy.get_param("~map_topic")
        self.num_beams_per_particle = rospy.get_param("~num_beams_per_particle")
        self.scan_theta_discretization = rospy.get_param("~scan_theta_discretization")
        self.scan_field_of_view = rospy.get_param("~scan_field_of_view")
        self.lidar_scale_to_map_scale = rospy.get_param("~lidar_scale_to_map_scale")

        # self.map_topic = "/map"
        # self.num_beams_per_particle = 100
        # self.scan_theta_discretization = 500
        # self.scan_field_of_view = 4.71
        # self.lidar_scale_to_map_scale = 1

        ####################################
        # TODO
        # Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201

        self.map_resolution = None
        ####################################

        # Precompute the sensor model table
        self.sensor_model_table = None
        self.precompute_sensor_model()

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
                self.num_beams_per_particle,
                self.scan_field_of_view,
                0, # This is not the simulator, don't add noise
                0.01, # This is used as an epsilon
                self.scan_theta_discretization) 

        # Subscribe to the map
        self.map = None
        self.map_set = False
        rospy.Subscriber(
                self.map_topic,
                OccupancyGrid,
                self.map_callback,
                queue_size=1)

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.
        
        For each discrete computed range value, this provides the probability of 
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A
        
        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """
        self.sensor_model_table = np.empty((self.table_width,self.table_width), dtype=np.float64)

        for d in range(self.table_width):
            phittable = np.empty(self.table_width)
            for zk in range(self.table_width):
                phittable[zk] = np.exp(-(zk-d)**2/(2.0*self.sigma_hit**2))/np.sqrt(2.0*np.pi*self.sigma_hit**2)
                pmax = 1 if zk == (self.table_width - 1) else 0
                pshort = 2*(1-zk*1.0/d)/d if (zk <= d and d != 0) else 0
                prand = 1.0/(self.table_width-1)
                self.sensor_model_table[zk][d] =  self.alpha_max*pmax + self.alpha_short*pshort + self.alpha_rand*prand
            phittable = phittable/(phittable.sum())
            self.sensor_model_table[:,d] = self.sensor_model_table[:,d] + phittable*self.alpha_hit
            self.sensor_model_table[:,d] = self.sensor_model_table[:,d]/(self.sensor_model_table[:,d].sum())
    
    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:
            
                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar.

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            return

        ####################################
        # TODO
        # Evaluate the sensor model here!
        #
        # You will probably want to use this function
        # to perform ray tracing from all the particles.
        # This produces a matrix of size N x num_beams_per_particle 

        z_max = self.table_width - 1
        ## REMOVE:
        # self.map_resolution = 0.0504
        meter_to_pixel = self.map_resolution*self.lidar_scale_to_map_scale

        scans = self.scan_sim.scan(particles)
        for par in range(len(scans)):
            scans[par] = [value*1.0/meter_to_pixel for value in scans[par]]
            scans[par] = [0 if value < 0 else z_max if value > z_max else value for value in scans[par]]

        # scale observation
        observation = [obs*1.0/meter_to_pixel for obs in observation]
        observation = [0 if obs < 0 else (z_max if obs > z_max else obs) for obs in observation]
    
        probabilities = np.zeros(len(scans))
        for par in range(len(scans)):
            log_prob = 0
            for beam in range(self.num_beams_per_particle):
                d = int(round(scans[par, beam])) # measured distance
                zk = int(round(observation[beam])) # ground-truth distance
                log_prob += np.log(self.sensor_model_table[zk][d])
            probabilities[par] = np.exp(log_prob)
            probabilities[par] = np.power(probabilities[par], 1/2.2)

        return probabilities
        ####################################

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double)/100.
        self.map = np.clip(self.map, 0, 1)
        self.map_resolution = map_msg.info.resolution
        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = tf.transformations.euler_from_quaternion((
                origin_o.x,
                origin_o.y,
                origin_o.z,
                origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
                self.map,
                map_msg.info.height,
                map_msg.info.width,
                map_msg.info.resolution,
                origin,
                0.5) # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        print("Map initialized")