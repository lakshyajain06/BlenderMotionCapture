import numpy as np

def get_class():
    print("hello")

class StereoNormalPointSolver:
    def __init__(self, offset, camera_resolution, fov):
        self.camera_offset = offset
        self.cam1_pos = np.array([-self.camera_offset / 2, 0, 0])
        self.cam2_pos = np.array([self.camera_offset / 2, 0, 0])
        self.pixel_width = camera_resolution[0]
        self.pixel_height = camera_resolution[1]
        self.fovX = np.radians(fov)
        self.projection_dist = self.get_projection_distance()

    def get_projection_distance(self):
        half_width = self.pixel_width / 2
        half_fov = self.fovX / 2

        d = half_width / np.tan(half_fov)
        return d
    
    def get_pixel_vector(self, x, y):
        x = x - (self.pixel_width / 2)
        y = y - ((self.pixel_width / 2))

        v = np.array([x, y, self.projection_dist])
        v = v / np.linalg.norm(v)

        return v
    
    def get_coordinate(self, cam1_coord, cam2_coord):
        pixel_vector1 = self.get_pixel_vector(cam1_coord[0], cam1_coord[1])
        pixel_vector2 = self.get_pixel_vector(cam2_coord[0], cam2_coord[1])

        A = np.array([
                [np.dot(pixel_vector1, pixel_vector1), -np.dot(pixel_vector2, pixel_vector1)], 
                [np.dot(pixel_vector1, pixel_vector2), -np.dot(pixel_vector2, pixel_vector2)]
            ])

        b = np.array([
                [np.dot((self.cam1_pos - self.cam2_pos), pixel_vector1)],
                [np.dot((self.cam1_pos - self.cam2_pos), pixel_vector2)]
            ])
        
        parametric_solution = np.linalg.solve(A, b)

        t1 = parametric_solution[0]
        t2 = parametric_solution[1]

        point1 = self.cam1_pos + (pixel_vector1 * t1)
        point2 = self.cam2_pos + (pixel_vector2 * t2)

        average_point = (point1 + point2) / 2

        return average_point
    
    def get_distance_from_cam(self, position, leftCam = True):
        if leftCam:
            distance = position - self.cam1_pos
        else:
            distance = position - self.cam2_pos

        distance = np.linalg.norm(distance)
        return distance