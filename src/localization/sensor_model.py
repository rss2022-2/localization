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

if __name__ == "__main__":
    sm = SensorModel()
    TEST_SENSOR_MODEL_INPUT_SCANS = [1.443311214447021484e+00, 1.524362444877624512e+00, 1.471921682357788086e+00, 1.412955999374389648e+00, 1.382264375686645508e+00, 1.316164731979370117e+00, 1.312758088111877441e+00, 1.254377126693725586e+00, 1.211054921150207520e+00, 1.208464026451110840e+00, 1.222940325736999512e+00, 1.183962225914001465e+00, 1.170150995254516602e+00, 1.144699573516845703e+00, 1.143659472465515137e+00, 1.138700246810913086e+00, 1.140269994735717773e+00, 1.138195753097534180e+00, 1.157852768898010254e+00, 1.182615637779235840e+00, 1.025218725204467773e+00, 7.623071074485778809e-01, 6.129682064056396484e-01, 6.105989813804626465e-01, 5.832239985466003418e-01, 5.764055252075195312e-01, 5.753004550933837891e-01, 6.456654667854309082e-01, 6.319560408592224121e-01, 6.494028568267822266e-01, 6.728968620300292969e-01, 6.781839132308959961e-01, 7.251351475715637207e-01, 7.653371691703796387e-01, 7.794520258903503418e-01, 1.777006626129150391e+00, 1.807145595550537109e+00, 1.725324630737304688e+00, 1.666449069976806641e+00, 1.651386976242065430e+00, 1.633908152580261230e+00, 1.573341012001037598e+00, 1.560045361518859863e+00, 1.530584096908569336e+00, 1.475903034210205078e+00, 1.479434013366699219e+00, 1.447488546371459961e+00, 1.438275456428527832e+00, 1.449833750724792480e+00, 1.437483549118041992e+00, 1.462784290313720703e+00, 1.155455589294433594e+00, 3.640684485435485840e-01, 2.726440727710723877e-01, 2.165609300136566162e-01, 2.184528559446334839e-01, 2.068135738372802734e-01, 2.168505191802978516e-01, 2.191669940948486328e-01, 2.172931730747222900e-01, 1.922705173492431641e-01, 2.200728356838226318e-01, 2.270477414131164551e-01, 2.002670466899871826e-01, 6.209039688110351562e-01, 6.653276085853576660e-01, 7.053702473640441895e-01, 7.698135972023010254e-01, 7.406820654869079590e-01, 7.902801036834716797e-01, 8.592942953109741211e-01, 9.427358508110046387e-01, 9.896806478500366211e-01, 1.096720457077026367e+00, 1.189790487289428711e+00, 1.314671277999877930e+00, 1.292042255401611328e+00, 1.273873567581176758e+00, 1.258950948715209961e+00, 1.240428686141967773e+00, 1.227135896682739258e+00, 1.224057674407958984e+00, 1.270466685295104980e+00, 1.197853446006774902e+00, 1.208770871162414551e+00, 1.224702119827270508e+00, 1.261736273765563965e+00, 1.231667876243591309e+00, 1.298240423202514648e+00, 1.307923316955566406e+00, 1.318225264549255371e+00, 1.307978987693786621e+00, 1.390820384025573730e+00, 1.358800768852233887e+00, 7.456571459770202637e-01, 7.621809840202331543e-01, 7.868509292602539062e-01, 7.815454006195068359e-01, 8.542019128799438477e-01, 9.016831517219543457e-01]
    TEST_PARTICLES_2 = [[ 0.56693784, -0.91253048, 0.0],
                    [ 1.51682839, -0.78764667, 0.0],
                    [-0.57632382,  0.70341235, 0.0],
                    [-0.16541293,  1.14213975, 0.0],
                    [-0.61713487,  0.30753082, 0.0],
                    [ 0.69595564,  0.6478497 , 0.0],
                    [ 0.5952413 ,  0.45499253, 0.0],
                    [ 1.58547838,  0.57026321, 0.0],
                    [ 1.59876167, -0.97415565, 0.0],
                    [-0.48491081,  0.7524957 , 0.0],
                    [-0.14234634, -0.96691631, 0.0],
                    [ 1.09743907,  0.86244721, 0.0],
                    [ 1.56330408,  0.88338316, 0.0],
                    [ 0.04969704, -0.27977549, 0.0],
                    [ 1.28758059,  0.33713363, 0.0],
                    [ 0.13241444, -0.72502182, 0.0],
                    [ 1.01717925,  0.65801722, 0.0],
                    [ 0.82417943,  1.12119327, 0.0],
                    [ 1.31135328, -0.06875884, 0.0],
                    [ 0.31515096,  0.51548214, 0.0],
                    [-0.05284796,  0.4801354 , 0.0],
                    [ 0.63911733, -0.93892983, 0.0],
                    [ 1.25252481,  0.18629184, 0.0],
                    [ 1.25157154,  0.69776309, 0.0],
                    [ 0.05070505,  0.49753587, 0.0],
                    [-0.14657836,  0.19314118, 0.0],
                    [ 0.33541871,  0.823377  , 0.0],
                    [ 0.46384173, -0.75930797, 0.0],
                    [ 0.77104009, -0.45711153, 0.0],
                    [-0.32435948, -0.28597683, 0.0],
                    [ 1.06355958, -0.52773234, 0.0],
                    [ 0.0679945 , -0.0042355 , 0.0],
                    [-0.30342478, -0.54080266, 0.0],
                    [ 1.40632412, -0.02578219, 0.0],
                    [-0.28977035,  0.26538201, 0.0],
                    [ 1.43657304, -0.64761899, 0.0],
                    [-0.17375377,  0.49522231, 0.0],
                    [ 0.58421734,  0.69543255, 0.0],
                    [-0.12493497,  0.90908753, 0.0],
                    [ 0.45792883,  1.16955355, 0.0],
                    [ 0.86248487,  0.32047137, 0.0],
                    [ 0.72686298,  0.47024011, 0.0],
                    [ 1.15928325, -0.31224229, 0.0],
                    [ 0.62738462,  0.43020033, 0.0],
                    [ 1.39964847,  0.47720918, 0.0],
                    [ 0.13808495,  1.00561365, 0.0],
                    [-0.64048591, -1.001921  , 0.0],
                    [-0.23568963, -0.45794166, 0.0],
                    [ 1.28366613, -0.89833606, 0.0],
                    [ 0.1760482 ,  0.69393116, 0.0]]
    print(sm.evaluate(TEST_PARTICLES_2, TEST_SENSOR_MODEL_INPUT_SCANS))