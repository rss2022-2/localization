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

        ####################################


    # @staticmethod
    # def __sample_normal(val, a):
    #     return val - val*np.random.normal(scale=a)

    @staticmethod
    def __sample_motion_model_odometry(position, odometry, a, deterministic = False):
        x = position[0]
        y = position[1]
        t = position[2]

        dx = odometry[0]
        dy = odometry[1]
        dt = odometry[2]
        
        if not deterministic:
            dx += dx*np.random.normal(scale=a[0])
            dy += dy*np.random.normal(scale=a[1])
            dt += dt*np.random.normal(scale=a[2])

        res = np.zeros(3)
        abs_dx = dx*np.cos(t) - dy*np.sin(t)
        abs_dy = dx*np.sin(t) + dy*np.cos(t)
        
        res[0] = x + abs_dx
        res[1] = y + abs_dy
        res[2] = t + dt
        return res
    
    @staticmethod
    def __add_noise(position, a):
        x = position[0] + np.random.normal(scale=a[0])
        y = position[1] + np.random.normal(scale=a[1])
        t = position[2] + np.random.normal(scale=a[2])
        
        return np.array([x, y, t])


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
        # TODO
        return np.array([MotionModel.__sample_motion_model_odometry(particle, odometry, self.a, self.deterministic) for particle in particles]) 

        ####################################

    
    def add_noise(self, particles, a):
        """
        """
        return np.array([self.__add_noise(particle, a) for particle in particles]) 
