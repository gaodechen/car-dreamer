import carla
import random
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

NUM_SIMULATIONS = 200  # Number of Monte Carlo simulations
EGO_SPEED_RANGE = (0, 10)  # Speed range in m/s for ego vehicle
TIME_STEP = 0.05  # Simulation time step in seconds
SIM_DURATION = 2  # Simulate for 5 seconds per run
COLLISION_DISTANCE = 2.5  # Threshold for collision detection in meters

def visualize_combined(xml_file, world, ego_transforms, bg_transforms, bg_speeds, 
                       save_path='combined_visualization.png', mode='lane'):
    """
    Visualize all events and path.
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # ===== Part 1: Plot Waypoints from XML =====
    tree = ET.parse(xml_file)
    root = tree.getroot()

    all_waypoints = []

    for route in root.findall('./route'):
        waypoints = []
        waypoints_elem = route.find('./waypoints')
        if waypoints_elem is not None:
            for position in waypoints_elem.findall('./position'):
                x = float(position.get('x'))
                y = float(position.get('y'))
                z = float(position.get('z'))
                waypoints.append([x, y, z])
        if waypoints:
            waypoints = np.array(waypoints)
            all_waypoints.append(waypoints)

    for i, waypoints in enumerate(all_waypoints):
        ax.plot(waypoints[:, 0], waypoints[:, 1], 'o-', label=f'Route {i+1} Waypoints', alpha=0.6, linewidth=2)
        ax.scatter(waypoints[0, 0], waypoints[0, 1], color='green', s=150, marker='*', label='Route Start' if i == 0 else "")
        ax.scatter(waypoints[-1, 0], waypoints[-1, 1], color='purple', s=150, marker='*', label='Route End' if i == 0 else "")

    # ===== Part 2: Plot Multi-Event Simulations =====
    print("Running Monte Carlo simulation and plotting trajectories...")

    safe_plotted, collision_plotted = False, False
    all_collision_rates = []

    for idx, (ego_transform, bg_transform, bg_speed) in enumerate(zip(ego_transforms, bg_transforms, bg_speeds)):
        collision_rate, bg_trajectories = run_monte_carlo_simulation(
            world, ego_transform, [bg_transform], [bg_speed], mode)

        all_collision_rates.append(collision_rate)

        for i in bg_trajectories:
            bg_x, bg_y = zip(*bg_trajectories[i])
            ax.plot(bg_x, bg_y, linestyle='--', color='gray', alpha=0.5, linewidth=1.5)
            ax.scatter(bg_x[0], bg_y[0], color='orange', marker='s', s=100, 
                       label="BG Start" if idx == 0 and i == 0 else "", zorder=4)

        for _ in range(3):  # few samples per event
            ego_speed = random.uniform(*EGO_SPEED_RANGE)
            ego_traj, collision = simulate_ego_trajectory(ego_transform, ego_speed, bg_trajectories)
            x, y = zip(*ego_traj)
            if collision:
                ax.plot(x, y, 'r', alpha=0.6, linewidth=1.5,
                        label="Ego (Collision)" if not collision_plotted else "")
                collision_plotted = True
            else:
                ax.plot(x, y, 'b', alpha=0.6, linewidth=1.5,
                        label="Ego (Safe)" if not safe_plotted else "")
                safe_plotted = True

        ego_loc = ego_transform.location
        ax.scatter(ego_loc.x, ego_loc.y, color='black', marker='o', s=150,
                   label="Ego Start" if idx == 0 else "", zorder=7)
        ax.text(ego_loc.x + 1.0, ego_loc.y + 1.0, f"{collision_rate:.1%}", fontsize=18,
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))

     # ===== Total Collision Rate Annotation =====
    total_collision_rate = 1.0
    for r in all_collision_rates:
        total_collision_rate *= (1 - r)
    total_collision_rate = 1.0 - total_collision_rate

    ax.text(0.98, 0.02, f"Total Collision Rate: {total_collision_rate:.2%}", 
        transform=ax.transAxes, fontsize=22, color='red', ha='right', va='bottom',
        bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.4', linewidth=2))

    
    # ===== Final Formatting =====
    ax.set_xlabel('X Position (m)', fontsize=18)
    ax.set_ylabel('Y Position (m)', fontsize=18)
    ax.set_title('Combined Visualization: Multi-Events Vehicle Trajectories', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1), fontsize=16)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Combined visualization saved as '{save_path}'")
    plt.show()

    return all_collision_rates




def get_lane_trajectory_from_transform(transform, world, num_points=100, step=2):
    """Retrieve waypoints along the lane for a given transformation."""
    map = world.get_map()
    traj = []
    waypoint = map.get_waypoint(transform.location)
    
    for _ in range(num_points):
        traj.append((waypoint.transform.location.x, waypoint.transform.location.y))
        next_waypoints = waypoint.next(step)
        if next_waypoints:
            waypoint = next_waypoints[0]
        else:
            break
            
    return traj

def precompute_background_trajectories_from_transforms(bg_transforms, bg_speeds, world, mode='lane'):
    """
    Precompute the motion trajectories for background vehicles based on their transformations.
    
    Parameters:
    -----------
    bg_transforms : list
        List of background vehicle transforms
    bg_speeds : list
        List of speeds for each background vehicle
    world : object
        CARLA world object
    mode : str
        'lane' - vehicles follow lane trajectory (default)
        'straight' - vehicles go straight in their initial direction
    """
    bg_trajectories = {}

    for i, transform in enumerate(bg_transforms):
        speed = bg_speeds[i]
        bg_trajectories[i] = []  # Using index instead of vehicle ID
        total_time_steps = int(SIM_DURATION / TIME_STEP)
        
        if mode == 'straight':
            # Simple mode: vehicles go straight
            # Get initial position
            start_x = transform.location.x
            start_y = transform.location.y
            
            # Get direction from transform rotation (yaw in degrees)
            yaw_rad = np.radians(transform.rotation.yaw)
            direction_x = np.cos(yaw_rad)
            direction_y = np.sin(yaw_rad)
            
            # Calculate trajectory points
            for t in range(total_time_steps):
                distance_traveled = speed * t * TIME_STEP
                bg_x = start_x + direction_x * distance_traveled
                bg_y = start_y + direction_y * distance_traveled
                bg_trajectories[i].append((bg_x, bg_y))
                
        else:
            # Original mode: follow lane trajectory
            traj = get_lane_trajectory_from_transform(transform, world)
            
            for t in range(total_time_steps):
                distance_traveled = speed * t * TIME_STEP
                cumulative_distance = 0.0
                bg_x, bg_y = traj[0]  # Default to start position

                for j in range(1, len(traj)):
                    prev_x, prev_y = traj[j - 1]
                    curr_x, curr_y = traj[j]
                    segment_distance = np.linalg.norm(np.array([curr_x, curr_y]) - np.array([prev_x, prev_y]))
                    cumulative_distance += segment_distance

                    if cumulative_distance >= distance_traveled:
                        ratio = (distance_traveled - (cumulative_distance - segment_distance)) / segment_distance
                        bg_x = prev_x + ratio * (curr_x - prev_x)
                        bg_y = prev_y + ratio * (curr_y - prev_y)
                        break

                bg_trajectories[i].append((bg_x, bg_y))

    return bg_trajectories

def simulate_ego_trajectory(ego_transform, ego_speed, bg_trajectories, mode='lane'):
    """Simulate ego vehicle motion and check for collision with precomputed background trajectories."""
    ego_x, ego_y = ego_transform.location.x, ego_transform.location.y
    ego_yaw = np.deg2rad(ego_transform.rotation.yaw)
    
    ego_traj = []
    collision = False
    total_time_steps = int(SIM_DURATION / TIME_STEP)

    for t in range(total_time_steps):
        ego_x += ego_speed * TIME_STEP * np.cos(ego_yaw)
        ego_y += ego_speed * TIME_STEP * np.sin(ego_yaw)
        ego_traj.append((ego_x, ego_y))

        # Check for collisions with background vehicles
        for i in bg_trajectories:
            bg_x, bg_y = bg_trajectories[i][t]
            if np.linalg.norm(np.array([ego_x, ego_y]) - np.array([bg_x, bg_y])) < COLLISION_DISTANCE:
                collision = True

    return ego_traj, collision

def run_monte_carlo_simulation(world, ego_transform, bg_transforms, bg_speeds, mode='lane'):
    """
    Run Monte Carlo simulations to compute the collision rate of the ego vehicle.
    """
    collision_count = 0
    
    bg_trajectories = precompute_background_trajectories_from_transforms(bg_transforms, bg_speeds, world, mode)
    
    for _ in range(NUM_SIMULATIONS):
        random_speed = random.uniform(*EGO_SPEED_RANGE)
        _, collision = simulate_ego_trajectory(ego_transform, random_speed, bg_trajectories)
        if collision:
            collision_count += 1

    collision_rate = collision_count / NUM_SIMULATIONS
    print(f"Collision rate: {collision_rate:.2%}")
    return collision_rate, bg_trajectories

def visualize_trajectories(world, ego_transform, bg_transforms, bg_speeds, mode='lane'):
    """Visualize Monte Carlo simulation results with background trajectories."""
    plt.figure(figsize=(12, 8))
    
    collision_rate, bg_trajectories = run_monte_carlo_simulation(world, ego_transform, bg_transforms, bg_speeds, mode)
    
    # Background vehicle trajectories
    for i in bg_trajectories:
        bg_x, bg_y = zip(*bg_trajectories[i])
        plt.plot(bg_x, bg_y, linestyle='--', color='gray', alpha=0.7)
        plt.scatter(bg_x[0], bg_y[0], color='purple', marker='x', s=120, label="BG Start" if i == 0 else "")
    
    safe_plotted, collision_plotted = False, False

    for _ in range(10):  # Plot only 10 trajectories for clarity
        ego_speed = random.uniform(*EGO_SPEED_RANGE)
        ego_traj, collision = simulate_ego_trajectory(ego_transform, ego_speed, bg_trajectories)
        x, y = zip(*ego_traj)

        if collision:
            plt.plot(x, y, 'r', alpha=0.5, label="Ego (Collision)" if not collision_plotted else "")
            collision_plotted = True
        else:
            plt.plot(x, y, 'b', alpha=0.5, label="Ego (Safe)" if not safe_plotted else "")
            safe_plotted = True
    
    plt.scatter(ego_transform.location.x, ego_transform.location.y, color='black', marker='o', s=120, label="Ego Start")

    plt.legend()
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Monte Carlo Trajectory Visualization")
    plt.grid()
    
    plt.savefig("collision_simulation.png", dpi=300)
    plt.show(block=False)




# bg_speeds = [20]
# transforms = [vehicle.get_transform() for vehicle in l]  # Extract transforms. Assume the first vehicle is the ego
# rate, _ = run_monte_carlo_simulation(world, transforms[0], transforms[1:], bg_speeds)
# visualize_trajectories(world, transforms[0], transforms[1:], bg_speeds)
