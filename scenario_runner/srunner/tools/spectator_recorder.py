import os
import datetime
import cv2
import numpy as np
import carla

class SpectatorRecorder:
    def __init__(self, world, ego_vehicle, output_path='./videos/', 
                 height=15.0, pitch=-90, resolution=(640, 320), fov=90):
        """
        Initialize the BEV recorder to capture footage from above the ego vehicle
        
        Args:
            world: CARLA world object
            ego_vehicle: The ego vehicle to follow
            output_path: Directory to save videos
            height: Height above the vehicle for the BEV camera
            pitch: Camera pitch angle (default -90 for direct top-down view)
            resolution: Video resolution (width, height)
            fov: Field of view of the camera
        """
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.height = height
        self.pitch = pitch
        
        # Create output directory if it doesn't exist
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_file = os.path.join(self.output_path, f'time{timestamp}.mp4')
        
        # Set up camera blueprint
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(resolution[0]))
        camera_bp.set_attribute('image_size_y', str(resolution[1]))
        camera_bp.set_attribute('fov', str(fov))
        
        # Create camera (not attached to any actor)
        transform = carla.Transform(carla.Location(x=0, y=0, z=0), carla.Rotation(pitch=self.pitch))
        self.camera = world.spawn_actor(camera_bp, transform)
        
        # Set up video writer
        self.video_writer = None
        self.recording = False
        self.latest_frame = None
        
        # Register callback to capture frames
        self.camera.listen(self._process_image)
    
    def _process_image(self, image):
        """Callback to process and store camera images"""
        # Convert CARLA raw image to OpenCV format
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        array = array[:, :, ::-1]  # Convert from BGRA to RGB
        
        # Store the latest frame
        self.latest_frame = array
        
        # Write to video if recording
        if self.recording and self.video_writer is not None:
            self.video_writer.write(array)
    
    def update(self):
        """Update the camera position to follow the ego vehicle"""
        if not self.ego_vehicle or not self.ego_vehicle.is_alive:
            return
            
        # Get ego vehicle's current transform
        ego_transform = self.ego_vehicle.get_transform()
        
        # Position the camera above the vehicle
        camera_location = ego_transform.location + carla.Location(z=self.height)
        camera_transform = carla.Transform(camera_location, carla.Rotation(pitch=self.pitch))
        
        # Update camera transform
        self.camera.set_transform(camera_transform)
        
        # Also update the spectator (for visualization in the CARLA client)
        spectator = self.world.get_spectator()
        spectator.set_transform(camera_transform)
    
    def start_recording(self):
        """Start recording the video"""
        if self.recording:
            return
            
        # Initialize video writer
        if self.latest_frame is not None:
            h, w = self.latest_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.output_file, 
                fourcc, 
                20.0,  # FPS
                (w, h)
            )
            self.recording = True
            print(f"Started recording BEV video to {self.output_file}")
        else:
            print("Cannot start recording - no frames received yet")
    
    def stop_recording(self):
        """Stop recording and save the video file"""
        if not self.recording:
            return
            
        self.recording = False
        
        # Release the video writer
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            print(f"Saved BEV video to {self.output_file}")
    
    def clean_up(self):
        """Destroy actors and clean up resources"""
        if self.recording:
            self.stop_recording()
            
        if self.camera and self.camera.is_alive:
            self.camera.stop()
            self.camera.destroy()
