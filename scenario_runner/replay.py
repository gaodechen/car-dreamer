import carla
import time
import os

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    # TODO: replace the actor ID here
    client.replay_file('./record/SignalizedJunctionLeftTurn_1.log', 0.0, 0.0, 194)

    world = client.get_world()
    
    actors = world.get_actors()
    print("All actors in the scene:")
    for actor in actors:
        print(f"  - Actor ID: {actor.id}, Type: {actor.type_id}")
    
    spectator_actor = None
    for actor in actors:
        if "spectator" in actor.type_id.lower():
            spectator_actor = actor
            break
    
    if spectator_actor is None:
        print("No spectator actor found in the actor list. Exiting.")
        return
    
    print(f"Found spectator actor: ID={spectator_actor.id}, Type={spectator_actor.type_id}")

    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')

    camera_transform = carla.Transform(carla.Location(x=0, y=0, z=2))

    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=spectator_actor)
    print(f"Attached camera (ID={camera.id}) to Spectator (ID={spectator_actor.id}).")

    os.makedirs('./images', exist_ok=True)

    def save_image(image):
        image.convert(carla.ColorConverter.Raw)
        filename = f"./images/{image.frame:06d}.png"
        image.save_to_disk(filename)
        print(f"Saved image: {filename}")

    camera.listen(lambda image: save_image(image))

    try:
        print("Streaming images from the spectator camera... Press Ctrl+C to stop.")
        while True:
            world.tick()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Interrupted by user. Cleaning up...")
    finally:
        camera.stop()
        camera.destroy()
        print("Camera destroyed.")

if __name__ == "__main__":
    main()