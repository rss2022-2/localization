import numpy as np

import rospy

class MotionModel:

    def __init__(self):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.
        
        self.deterministic = rospy.get_param("~deterministic", False)
        ax = rospy.get_param("~motion_model_ax", 0.025)
        ay = rospy.get_param("~motion_model_ay", 0.025)
        at = rospy.get_param("~motion_model_at", 0.005)
        self.a = [ax, ay, at]
        self.local_deltas = None

        ####################################

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:
            
                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """
        
        ####################################
        N = particles.shape[0]
        cosines = np.cos(particles[:,2])
        sines = np.sin(particles[:,2])

        if self.local_deltas is None: self.local_deltas = np.zeros((N,3))
        self.local_deltas[:,0] = cosines*odometry[0] - sines*odometry[1]
        self.local_deltas[:,1] = sines*odometry[0] + cosines*odometry[1]
        self.local_deltas[:,2] = odometry[2]

        particles[:,:] += self.local_deltas

        if not self.deterministic:
            particles[:,0] += odometry[0]*np.random.normal(loc=0.0,scale=self.a[0],size=N)
            particles[:,1] += odometry[1]*np.random.normal(loc=0.0,scale=self.a[1],size=N)
            particles[:,2] += odometry[2]*np.random.normal(loc=0.0,scale=self.a[2],size=N) + np.random.normal(loc=0.0,scale=self.a[2],size=N)/3

        return particles

        ####################################

    
    def add_noise(self, particles, a):
        """
        """
        N = particles.shape[0]
        particles[:,0] += np.random.normal(loc=0.0,scale=a[0],size=N)
        particles[:,1] += np.random.normal(loc=0.0,scale=a[1],size=N)
        particles[:,2] += np.random.normal(loc=0.0,scale=a[2],size=N)

        return particles
        