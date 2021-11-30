#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

''' This file comes with CARLA as an example to use with the simulator.
    Title: CARLA synchronous_mode.py
    See Copyright Above
    Availability: https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/synchronous_mode.py
'''



import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
sys.path.append(r'C:\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.10-py3.7-win-amd64.egg')
import carla

import random

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def main():
    actor_list = []
    x_camera = 1.5
    y_camera = 0
    z_camera = 2
    x_lidar = -5
    y_lidar = 0
    z_lidar = 7
    pygame.init()

    display = pygame.display.set_mode(
        (640, 480),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(100.0)

    world = client.get_world()

    try:
        m = world.get_map()
        start_pose = random.choice(m.get_spawn_points())
        waypoint = m.get_waypoint(start_pose.location)

        blueprint_library = world.get_blueprint_library()
        vehicle_name = blueprint_library.find('vehicle.ford.mustang')
        #vehicle_name.set_attribute('set_target_velocity', '4.47')

        vehicle = world.spawn_actor(vehicle_name,start_pose)
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(True)
        velocity_mustang = carla.Vector3D(4.47,0,0)
        vehicle.set_target_velocity(velocity_mustang)

        lidar_sensor = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_sensor.set_attribute('rotation_frequency', '10')
        lidar_sensor.set_attribute('points_per_second', '8000')
        lidar_sensor.set_attribute('range', '10')
        print(lidar_sensor.get_attribute('rotation_frequency'))

        lidar3d = world.spawn_actor(lidar_sensor,
            carla.Transform(carla.Location(x_lidar, y_lidar, z_lidar), carla.Rotation(pitch=360)),
            attach_to=vehicle)
        actor_list.append(lidar3d)

        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x_camera, y_camera, z_camera), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)

        camera_semseg = world.spawn_actor(
            blueprint_library.find('sensor.camera.semantic_segmentation'),
            carla.Transform(carla.Location(x_camera, y_camera, z_camera), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_semseg)

        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, camera_semseg,lidar3d, fps=10) as sync_mode:
            while True:
                if should_quit():
                    return
                clock.tick()

                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_semseg, lidar_raycast = sync_mode.tick(timeout=100.0)

                # Choose the next waypoint and update the car location.
                
                waypoint = random.choice(waypoint.next(1))
                waypoint2 = random.choice(waypoint.next(2))
                waypoint3 = random.choice(waypoint.next(3))
                waypoint4 = random.choice(waypoint.next(4))
                waypoint5 = random.choice(waypoint.next(5))
                waypoint6 = random.choice(waypoint.next(6))
                waypoint7 = random.choice(waypoint.next(7))
                waypoint8 = random.choice(waypoint.next(8))
                waypoint9 = random.choice(waypoint.next(9))
                waypoint10 = random.choice(waypoint.next(10))

                vehicle.set_transform(waypoint.transform)
                print(waypoint)
                print(waypoint2)
                print(waypoint3)
                print(waypoint4)
                print('donewithone')
                image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                fps = round(1.0 / snapshot.timestamp.delta_seconds)
                # Draw the display.
                draw_image(display, image_rgb)
                #draw_image(display, image_semseg, blend=True)
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                pygame.display.flip()
                lidar_raycast.save_to_disk('C:/Tommy/SDP33/LiDAR')
                image_rgb.save_to_disk('C:/Tommy/SDP33/camera')
                

    finally:

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
