import numpy as np

class MotionModel:

    def __init__(self):

        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.
        self.a = [
                    0,  # ax
                    0,  # ay
                    0,  # at
                 ] 

        ####################################


    @staticmethod
    def __sample_normal(val, a):
        return val - a*val*np.random.standard_normal()

    def __sample_motion_model_odometry(self, position, odometry):
        x = position[0]
        y = position[1]
        t = position[2]

        dx = odometry[0]
        dy = odometry[1]
        dt = odometry[2]

        res = [0, 0, 0]
        abs_dx = dx*np.cos(t) - dy*np.sin(t)
        abs_dy = dx*np.sin(t) + dy*np.cos(t)
        res[0] = x + MotionModel.__sample_normal(abs_dx, self.a[0])
        res[1] = y + MotionModel.__sample_normal(abs_dy, self.a[1])
        res[2] = t + MotionModel.__sample_normal(dt, self.a[2])
        return res


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
        return [self.__sample_motion_model_odometry(particle, odometry) for particle in particles] 

        ####################################