import glob
import os
import sys
from tensorflow.keras import models
import random
import numpy as np
import pygame
from synchronous_mode import CarlaSyncMode, get_font, should_quit, draw_image
from pure_pursuit import PurePursuitPlusPID
from helper_functions import *

sys.path.append(r'C:\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.10-py3.7-win-amd64.egg')
import carla

model_name = r'transfer_model.h5'


def main():

    actor_list = []
    pygame.init()

    display = pygame.display.set_mode(
        (640, 480),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(80.0)

    client.load_world('Town04')
    world = client.get_world()
    weather_presets = find_weather_presets()
    world.set_weather(weather_presets[3][0])

    controller = PurePursuitPlusPID()
    
    model = models.load_model(model_name)
    
    try:
        m = world.get_map()

        blueprint_library = world.get_blueprint_library()

        veh_bp = random.choice(blueprint_library.filter('vehicle.audi.tt'))
        veh_bp.set_attribute('color','64,81,181')
        vehicle = world.spawn_actor(
            veh_bp,
            m.get_spawn_points()[90])
        actor_list.append(vehicle)


        # rgb cam (training/prediction functionality)
        camera_bp =  blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x",'640')  #set camera Width
        camera_bp.set_attribute("image_size_y",'480') #set camera height
        camera_bp.set_attribute("fov",'62.2')
        camera_bp.set_attribute("fstop",'2')
        camera_rgb = world.spawn_actor(camera_bp,carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)
        sensors = [camera_rgb]
        
        # visualization cam (no functionality with angle directly from behind)
        visualization_bp =  blueprint_library.find('sensor.camera.rgb')
        visualization_bp.set_attribute("image_size_x",'640')  #set camera Width
        visualization_bp.set_attribute("image_size_y",'480') #set camera height
        visualization_bp.set_attribute("fov",'62.2')
        visualization_bp.set_attribute("fstop",'2')
        visualization_cam = world.spawn_actor(visualization_bp,carla.Transform(carla.Location(x=-7.5,y=0, z=3), carla.Rotation(pitch=-15, yaw=0)),
            attach_to=vehicle)
        actor_list.append(visualization_cam)
        sensors.append(visualization_cam)
        
        # visualization cam (no functionality with angle from left behing)
        visualization_left_bp =  blueprint_library.find('sensor.camera.rgb')
        visualization_left_bp.set_attribute("image_size_x",'640')  #set camera Width
        visualization_left_bp.set_attribute("image_size_y",'480') #set camera height
        visualization_left_bp.set_attribute("fov",'62.2')
        visualization_left_bp.set_attribute("fstop",'2')
        visualization_cam_left = world.spawn_actor(visualization_bp,carla.Transform(carla.Location(x=-7.5,y=-2, z=3), carla.Rotation(pitch=-15, yaw=15)),
            attach_to=vehicle)
        actor_list.append(visualization_cam_left)
        sensors.append(visualization_cam_left)
        
        # visualization cam (no functionality with angle from right behind)
        visualization_right_bp =  blueprint_library.find('sensor.camera.rgb')
        visualization_right_bp.set_attribute("image_size_x",'640')  #set camera Width
        visualization_right_bp.set_attribute("image_size_y",'480') #set camera height
        visualization_right_bp.set_attribute("fov",'62.2')
        visualization_right_bp.set_attribute("fstop",'2')
        visualization_cam_right = world.spawn_actor(visualization_bp,carla.Transform(carla.Location(x=-7.5,y=2, z=3), carla.Rotation(pitch=-15, yaw=-15)),
            attach_to=vehicle)
        actor_list.append(visualization_cam_right)
        sensors.append(visualization_cam_right)


        frame = 0
        max_error = 0
        FPS = 30
        
        
        # Create a synchronous mode context.
        with CarlaSyncMode(world, *sensors, fps=FPS) as sync_mode:
            while frame <= 4000: #equal to one loop around the track
                if should_quit():
                    return
                clock.tick()          
                
                tick_response = sync_mode.tick(timeout=2.0)
                #retrieve sensor data from vehicle to pass to network and display
                snapshot, image_rgb, visualization_image_rgb, visualization_image_left_rgb, visualization_image_right_rgb = tick_response

                traj_prediction = []
                img = np.array((process_img(image_rgb,False)).reshape(480,640,3))[100:300,120:520]
                #retrieve predicted reference trajectory from model
                prediction = model.predict(img.reshape(-1,200,400,3).reshape(1,1,200,400,3))[0]
                #reformat to the shape used for the motion planner
                traj_prediction = np.array([[float(x), prediction[x]]for x in range(len(prediction))])

                velocity = vehicle.get_velocity()
                speed = np.linalg.norm(np.array([velocity.x,velocity.y,velocity.z]))
                #retrieve vehicle controls from motion planner
                throttle, steer = controller.get_control(traj_prediction, speed, desired_speed=15, dt=1./FPS)
                send_control(vehicle, throttle, steer, 0)


                #calculate error and draw image to pygame frame    
                fps = round(5.0 / snapshot.timestamp.delta_seconds)
        
                dist = dist_point_linestring(np.array([0,0]), traj_prediction)

                cross_track_error = int(dist*100)
                max_error = max(max_error, cross_track_error)

                # Draw the display.
                if (frame // 300) % 3 == 0: #these if statements shift the camera angle
                    draw_image(display, visualization_image_rgb)
                elif (frame // 300) % 3 == 1:
                    draw_image(display,visualization_image_left_rgb)
                else:
                    draw_image(display,visualization_image_right_rgb)
                display.blit(
                    font.render('     FPS (real) % 5d ' % clock.get_fps(), True, (0,0,0)),
                    (8, 10))
                display.blit(
                    font.render('     FPS (simulated): % 5d ' % fps, True, (0,0,0)),
                    (8, 28))
                display.blit(
                    font.render('     speed: {:.2f} m/s'.format(speed), True, (0,0,0)),
                    (8, 46))
                display.blit(
                    font.render('     cross track error: {:03d} cm'.format(cross_track_error), True, (0,0,0)),
                    (8, 64))
                display.blit(
                    font.render('     max cross track error: {:03d} cm'.format(max_error), True, (0,0,0)),
                    (8, 82))

                pygame.display.flip()

                frame += 1
                

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
