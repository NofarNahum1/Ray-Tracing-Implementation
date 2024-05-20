from helper_classes import *
import numpy as np
import matplotlib.pyplot as plt


def render_scene(camera, ambient, lights, objects, screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

    image = np.zeros((height, width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            # screen is on origin
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalize(pixel - origin)
            ray = Ray(origin, direction)
            color = np.zeros(3)
            color = calculate_color(camera, ray, ambient, lights, objects, max_depth)
            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color, 0, 1)

    return image


def calculate_color(camera, ray, ambient, lights, objects, max_depth):
    if max_depth <= 0:
        return np.zeros(3)

    color = np.zeros(3)
    ambient_color = np.zeros(3)
    min_distance, nearest_object = ray.nearest_intersected_object(objects)

    if nearest_object is None:
        return color
    else:
        # Calculate ambient color
        ambient_color = nearest_object.ambient * ambient
        color += ambient_color
        intersection_point = ray.origin + min_distance * ray.direction
        # calc the diffuse color and the specular color:
        diffuse_color, specular_color = calculate_diffuse_specular_color(nearest_object, lights, intersection_point,
                                                                         objects, camera)
        color += (diffuse_color + specular_color)
        # calc the intersection of the refractive and reflective rays with the objects
        normal = None
        # isinstance function(built-in) checks if an object is an instance of a specified class
        if isinstance(nearest_object, Sphere):
            normal = normalize(intersection_point - nearest_object.center)
        else:
            normal = nearest_object.normal

        reflected_ray = calculate_reflected_ray(intersection_point, normal, ray)
        min_distance_to_ref, nearest_object_to_ref = reflected_ray.nearest_intersected_object(objects)

        if (nearest_object_to_ref is not None) and (nearest_object_to_ref.reflection != 0):
            # color += nearest_object.reflection * calculate_color(camera, reflected_ray, objects, lights, ambient, max_depth - 1)
            color += nearest_object.reflection * calculate_color(camera, reflected_ray, ambient, lights, objects,
                                                                 max_depth - 1)

        # Combine the ambient, diffuse, specular, and reflection colors to get the final pixel color.
        return color


def is_shadowed(intersection_point, intersected_object, objects, light) -> bool:
    light_ray = light.get_light_ray(intersection_point)
    max_distance = light.get_distance_from_light(intersection_point)

    for obj in objects:
        if obj != intersected_object:  # Exclude the object the ray intersected with
            intersection = obj.intersect(light_ray)
            if intersection is not None and intersection[0] <= max_distance:
                return True
    return False


def calculate_reflected_ray(intersection_point, normal, incoming_ray):
    shifting_factor = 0.001  # slightly move the intersection point along the surface normal
    direction = reflected(incoming_ray.direction, normal)
    shifted_hitP = intersection_point + shifting_factor * normal
    reflected_ray = Ray(shifted_hitP, direction)
    return reflected_ray


def compute_specular(light_dir, view_dir, normal, light_intensity, specular_color, shininess):
    reflect_dir = reflected(-light_dir, normal)
    spec = np.dot(view_dir, reflect_dir) ** shininess
    
    return specular_color * light_intensity * spec


def calculate_diffuse_specular_color(nearest_object, lights, intersection_point, objects, camera):
    normal = None
    diffuse_color = np.zeros(3)
    specular_color = np.zeros(3)

    if isinstance(nearest_object, Sphere):
        normal = normalize(intersection_point - nearest_object.center)
    else:
        normal = nearest_object.normal

    for light in lights:
        light_ray = light.get_light_ray(intersection_point)
        light_intensity = light.get_intensity(intersection_point)
        distance_from_light = light.get_distance_from_light(intersection_point)

        if is_shadowed(intersection_point, nearest_object, objects, light):
            continue  # skips the rest of the current iteration of the loop
        # calc the diffuse color:
        # diffuse_intensity = max(np.dot(normal, light_ray.direction), 0)
        diffuse_intensity = np.dot(normal, light_ray.direction)
        diffuse_color += nearest_object.diffuse * light_intensity * diffuse_intensity
        # calc the specular color:
        view_dir = normalize(camera - intersection_point)
        specular_color += compute_specular(light_ray.direction, view_dir, normal, light_intensity, nearest_object.specular, nearest_object.shininess)

    return diffuse_color, specular_color


# Write your own objects and lights
def your_own_scene():
    camera = np.array([0, 0, 1])
    lights = []
    
    light_a = SpotLight(intensity=np.array([0.8, 0.3, 0.8]), position=np.array([0.5, 0.5, 0]), direction=np.array([0, 0, -1]),
                        kc=0.1, kl=0.1, kq=0.1)
    lights.append(light_a)
    
    light_b = PointLight(intensity=np.array([1, 1, 1]), position=np.array([1, 1.5, 1]), kc=0.1, kl=0.1, kq=0.1)
    lights.append(light_b)  # Append light_b instead of light_a
    
    background = Plane(normal=[0, 0, 1], point=[0, 0, -1])
    background.set_material(ambient=[1, 1, 1], diffuse=[1, 1, 1], specular=[5, 5, 5], shininess=1000, reflection=0.5)

    # Heart shape in the middle of the image
    heart_sphere1 = Sphere(center=[-0.2, 0.2, -0.5], radius=0.3)
    heart_sphere1.set_material([1, 0, 0], [1, 0, 0], [1, 0.5, 0.5], 10, 0.5)

    heart_sphere2 = Sphere(center=[0.2, 0.2, -0.5], radius=0.3)
    heart_sphere2.set_material([1, 0, 0], [1, 0, 0], [1, 0.5, 0.5], 10, 0.5)

    # Create two triangles for the lower part of the heart
    heart_triangle1 = Triangle(a=[-0.2, -0.1, -0.5], b=[0.2, -0.1, -0.5], c=[0, -0.4, -0.5])
    heart_triangle1.set_material([1, 0, 0], [1, 0, 0], [1, 0.5, 0.5], 10, 0.5)

    heart_triangle2 = Triangle(a=[0, -0.4, -0.5], b=[-0.2, -0.1, -0.5], c=[0.2, -0.1, -0.5])
    heart_triangle2.set_material([1, 0, 0], [1, 0, 0], [1, 0.5, 0.5], 10, 0.5)


    objects = [background, heart_sphere1, heart_sphere2, heart_triangle1, heart_triangle2]

    return camera, lights, objects
