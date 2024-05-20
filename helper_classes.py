import numpy as np

# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)

# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, axis):
    normed_axis = normalize(axis)
    projection = np.dot(vector, normed_axis) * normed_axis
    return vector - 2 * projection


## Lights


class LightSource:
    def __init__(self, intensity):
        self.intensity = intensity


class DirectionalLight(LightSource):
    def __init__(self, intensity, direction):
        super().__init__(intensity)
        self.direction = normalize(np.array(direction))

    # This function returns the ray that goes from a point to the light source.
    def get_light_ray(self,intersection_point):
        return Ray(intersection_point, (-1) * normalize(self.direction))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        # there is no position to the sun/direct light hence the distance is unlimite 
        return np.inf

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        # I_L = I_0
        return self.intensity


class PointLight(LightSource):
    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from a point to the light source
    def get_light_ray(self,intersection):
        return Ray(intersection, normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self,intersection):
        return np.linalg.norm(intersection - self.position)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        return self.intensity / (self.kc + self.kl*d + self.kq * (d**2))  # I_0 / f_att


class SpotLight(LightSource):
    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.direction = normalize(direction)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from a point to the light source.
    def get_light_ray(self, intersection):
        return Ray(intersection, normalize(self.position - intersection))

    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)

    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        v = normalize(intersection - self.position)
        v_d = normalize(self.direction)
        gama = np.dot(v, v_d)
        if gama > 0:  # assuming the spotlight only affects the forward direction
            return (self.intensity * gama) / (self.kc + self.kl * d + self.kq * (d**2))  # = I_L
        else:
            return np.zeros(3) # No light if the angle is outside the spotlight cone

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    # The function is getting the collection of objects in the scene and looks for the one with minimum distance.
    # The function should return the nearest object and its distance (in two different arguments)
    def nearest_intersected_object(self, objects):
        intersections = []
        # Create a list of intersections by checking each object for an intersection with the ray
        for obj in objects:
            intersection = obj.intersect(self)
            if intersection is not None:
                intersections.append(intersection)
        
        # Filter out None values from the list of intersections
        valid_intersections = list(filter(None, intersections))
        # If there are valid intersections, find the nearest one
        if valid_intersections:
            nearest_intersection = min(valid_intersections, key=lambda x: x[0])  # x[0] is the distance t
        else:
            nearest_intersection = (np.inf, None)  # No intersection found

        return nearest_intersection # (distcnace, object)


class Object3D:
    def set_material(self, ambient, diffuse, specular, shininess, reflection):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection
        

class Plane(Object3D):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)

    def intersect(self, ray: Ray):
        v = self.point - ray.origin
        t = np.dot(v, self.normal) / (np.dot(self.normal, ray.direction) + 1e-6)  # t is the distance along the ray to the intersection point.
        if t > 0:
            return t, self
        else:
            return None

    def compute_normal(self, intersection_point):
        return normalize(self.normal)

class Triangle(Object3D):
    """
        C
        /\
       /  \
    A /____\ B

    The fornt face of the triangle is A -> B -> C.
    
    """
    def __init__(self, a, b, c):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.normal = self.compute_normal()

    # computes normal to the trainagle surface. Pay attention to its direction!
    def compute_normal(self):
        edge1 = self.b - self.a
        edge2 = self.c - self.a
        # Compute the normal using the cross product
        normal = np.cross(edge1, edge2)
        normalized_normal = normalize(normal)
        return normalized_normal

    def intersect(self, ray: Ray):
        edge1 = self.b - self.a
        edge2 = self.c - self.a
        epsilon = 1e-6

        # Calculate the vector perpendicular to the ray direction and edge2
        perpendicular_vector = np.cross(ray.direction, edge2)
        # Calculate a scalar which helps determine if the ray is parallel to the triangle
        determinant = np.dot(edge1, perpendicular_vector)

        if -epsilon < determinant < epsilon:  # If determinant is close to 0, the ray is parallel to the triangle and there is no intersection
            return None
        
        inverse_determinant = 1.0 / determinant
        origin_to_vertex = ray.origin - self.a
        u = inverse_determinant * np.dot(origin_to_vertex,perpendicular_vector) 

        if u < 0.0 or u > 1.0:  # Checks if the intersection point is outside the triangle
            return None
        
        q = np.cross(origin_to_vertex, edge1)
        v = inverse_determinant * np.dot(ray.direction, q)

        if v < 0.0 or u + v > 1.0:
            return None
        
        # t represents the distance from the ray origin to the intersection point
        t = inverse_determinant * np.dot(edge2, q)

        if t > epsilon:
            return t, self
        
        return None

    
class Pyramid(Object3D):
    """     
            D
            /\*\
           /==\**\
         /======\***\
       /==========\***\
     /==============\****\
   /==================\*****\
A /&&&&&&&&&&&&&&&&&&&&\ B &&&/ C
   \==================/****/
     \==============/****/
       \==========/****/
         \======/***/
           \==/**/
            \/*/
             E 
    
    Similar to Traingle, every from face of the diamond's faces are:
        A -> B -> D
        B -> C -> D
        A -> D -> C
        E -> B -> A
        E -> C -> B
        C -> E -> A
    """
    def __init__(self, v_list):
        self.v_list = v_list
        self.triangle_list = self.create_triangle_list()

    # Creates and returns a list of Triangle objects that form the faces of the pyramid
    def create_triangle_list(self):
        l = []
        t_idx = [
                [0,1,3],
                [1,2,3],
                [0,3,2],
                [4,1,0],
                [4,2,1],
                [2,4,0]]

        for indices in t_idx:
            a, b, c = [self.v_list[i] for i in indices]
            l.append(Triangle(a, b, c))

        return l

    def apply_materials_to_triangles(self):
        for triangle in self.triangle_list:
            triangle.set_material(self.ambient, self.diffuse, self.specular, self.shininess, self.reflection)

    def intersect(self, ray: Ray):
        min_distance = np.inf
        nearest_object = None

        for triangle in self.triangle_list:
            intersection = triangle.intersect(ray)
            if intersection is not None: 
                distance, _ = intersection
                if distance < min_distance:
                    min_distance = distance
                    nearest_object = triangle

        if nearest_object is None:
            return None
        else:
            return min_distance, nearest_object
    

class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray):
        a = np.linalg.norm(ray.direction)
        b = 2 * np.dot(ray.direction, ray.origin - self.center)
        c = np.linalg.norm(ray.origin - self.center) ** 2 - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            return None  # no intersection

        t1 = (- b + np.sqrt(discriminant)) / (2 * a)
        t2 = (- b - np.sqrt(discriminant)) / (2 * a)

        if t1 > 0 and t2 > 0:
            return np.min((t1, t2)), self
        elif t1 > 0 or t2 > 0:
            return np.max((t1, t2)), self
        return None


    def compute_normal(self, p):
        norm = normalize(p - self.center)
        plane = Plane(norm, p)
        return plane.compute_normal(p)