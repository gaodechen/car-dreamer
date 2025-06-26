import random
import argparse
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import carla
import sys
sys.path.append('..')
from srunner.tools.route_parser import RouteParser
from srunner.tools.route_manipulation import interpolate_trajectory
from srunner.tools.scenario_parser import ScenarioConfigurationParser
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from mc_simulation import run_monte_carlo_simulation, visualize_trajectories, visualize_combined
import random
import numpy as np
import math 

DIST_THRESHOLD = 2.0
ANGLE_THRESHOLD = 10

import matplotlib.pyplot as plt
import numpy as np

def extract_scenario_inputs(xml_file, world, bg_speed=2.0):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    ego_transforms = []
    bg_transforms = []
    bg_speeds = []

    for route in root.findall('./route'):
        for scenarios in route.findall('./scenarios'):
            for scenario in scenarios.findall('./scenario'):
                trigger_elem = scenario.find('./trigger_point')
                if trigger_elem is None:
                    continue

                # --- Ego Transform ---
                ego_loc = carla.Location(
                    x=float(trigger_elem.get('x')),
                    y=float(trigger_elem.get('y')),
                    z=float(trigger_elem.get('z', 0.0))
                )
                yaw = float(trigger_elem.get('yaw'))
                ego_rot = carla.Rotation(yaw=yaw)
                ego_transform = carla.Transform(ego_loc, ego_rot)

                # --- Parameters ---
                direction = scenario.get('direction', 'right')
                crossing_angle = float(scenario.get('crossing_angle', '0'))

                distance_elem = scenario.find('./distance')
                forward_distance = float(distance_elem.get('value')) if distance_elem is not None else 12.0

                # --- Get forward waypoint ---
                wmap = world.get_map()
                waypoint = wmap.get_waypoint(ego_loc)
                remaining_dist = forward_distance
                while remaining_dist > 0:
                    next_wps = waypoint.next(remaining_dist)
                    if not next_wps:
                        break
                    waypoint = next_wps[0]
                    remaining_dist = 0

                sidewalk_wp = waypoint

                # --- Lateral offset ---
                offset_dist = 0.5
                if direction == "left":
                    offset_dist *= -1

                lateral_vec = sidewalk_wp.transform.get_right_vector()
                base_loc = sidewalk_wp.transform.location + carla.Location(
                    x=lateral_vec.x * offset_dist,
                    y=lateral_vec.y * offset_dist,
                    z=1.2
                )

                # --- Rotation from lateral vector ---
                if direction == "right":
                    crossing_vec = carla.Vector3D(-lateral_vec.x, -lateral_vec.y, -lateral_vec.z)
                else:
                    crossing_vec = lateral_vec

                crossing_yaw = math.degrees(math.atan2(crossing_vec.y, crossing_vec.x))
                crossing_yaw += crossing_angle
                bg_rotation = carla.Rotation(yaw=crossing_yaw)

                bg_transform = carla.Transform(base_loc, bg_rotation)

                # --- Append results ---
                ego_transforms.append(ego_transform)
                bg_transforms.append(bg_transform)
                bg_speeds.append(bg_speed)

    return ego_transforms, bg_transforms, bg_speeds

def visualize_from_file(xml_file, save_path='visualization.png'):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    all_waypoints = []
    trigger_points = []

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

        for scenarios in route.findall('./scenarios'):
            for scenario in scenarios.findall('./scenario'):
                for trigger_point in scenario.findall('./trigger_point'):
                    x = float(trigger_point.get('x'))
                    y = float(trigger_point.get('y'))
                    trigger_points.append([x, y])

    trigger_points = np.array(trigger_points) if trigger_points else np.empty((0, 2))

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot waypoints
    for i, waypoints in enumerate(all_waypoints):
        ax.plot(waypoints[:, 0], waypoints[:, 1], 'o-', label=f'Route {i+1} Waypoints', alpha=0.6)

        # Plot start and end points with distinct markers and legend
        ax.scatter(waypoints[0, 0], waypoints[0, 1], color='green', s=100, marker='*', label='Start Point' if i == 0 else "")
        ax.scatter(waypoints[-1, 0], waypoints[-1, 1], color='purple', s=100, marker='*', label='End Point' if i == 0 else "")

    # Plot trigger points
    if len(trigger_points) > 0:
        ax.scatter(trigger_points[:, 0], trigger_points[:, 1], color='red', s=80, zorder=5, label='Mutated Trigger Points')
        for i, (x, y) in enumerate(trigger_points):
            ax.text(x, y, f'TP{i+1}', fontsize=9, ha='right')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Waypoint Paths and Mutated Scenario Trigger Points')
    ax.legend()
    ax.grid(True)
    plt.axis('equal')

    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved as '{save_path}'.")
    plt.show()

def compute_path_lengths_locations(waypoints):
    seg_lengths = [p1.distance(p2) for p1, p2 in zip(waypoints[:-1], waypoints[1:])]
    total_length = sum(seg_lengths)
    total_length = np.sum(seg_lengths)
    return total_length, seg_lengths

def get_route(config):
    """
    Gets the route from the configuration, interpolating it to the desired density,
    saving it to the CarlaDataProvider and sending it to the agent

    Parameters:
    - world: CARLA world
    - config: Scenario configuration (RouteConfiguration)
    - debug_mode: boolean to decide whether or not the route poitns are printed
    """
    # prepare route's trajectory (interpolate and add the GPS route)
    gps_route, route = interpolate_trajectory(config.keypoints)
    if config.agent is not None:
        config.agent.set_global_plan(gps_route, route)

    return route

def add_noise(value, noise_level=0.3):
    noise = random.uniform(-noise_level, noise_level)
    return round(float(value) + noise, 2)

def is_scenario_at_route(trigger_transform, route):
    """
    Check if the scenario is affecting the route.
    This is true if the trigger position is very close to any route point
    """
    
    def sample_valid_yaw(route_yaw, angle_threshold=10):
        delta = random.uniform(-angle_threshold, angle_threshold)
        valid_yaw = (route_yaw + delta) % 360
        angle_dist = (valid_yaw - route_transform.rotation.yaw) % 360
        if angle_dist < ANGLE_THRESHOLD or angle_dist > (360 - ANGLE_THRESHOLD):
            return valid_yaw
        else:
            print("not valid yaw")
    
    def is_trigger_close(trigger_transform, route_transform):
        """Check if the two transforms are similar"""
        dist = trigger_transform.location.distance(route_transform.location)
        if dist < DIST_THRESHOLD:
            return True

    for route_transform, _ in route:
        if is_trigger_close(trigger_transform, route_transform):
            yaw = sample_valid_yaw(route_transform.rotation.yaw)
            trigger_transform.rotation.yaw = yaw
            return trigger_transform

    return False

def add_random_trigger_points_along_path(waypoints, route, num_points=5, max_attempts=1000):
    
    total_length, seg_lengths = compute_path_lengths_locations(waypoints)
    seg_cumsum = np.cumsum(seg_lengths)

    generated_trigger_points = []
    attempts = 0

    while len(generated_trigger_points) < num_points and attempts < max_attempts:
        dist = np.random.uniform(0, total_length)
        seg_idx = np.searchsorted(seg_cumsum, dist)
        seg_start = 0 if seg_idx == 0 else seg_cumsum[seg_idx - 1]
        t = (dist - seg_start) / seg_lengths[seg_idx]

        p1 = waypoints[seg_idx]
        p2 = waypoints[seg_idx + 1]
        # Interpolate
        x = (1 - t) * p1.x + t * p2.x
        y = (1 - t) * p1.y + t * p2.y
        z = (1 - t) * p1.z + t * p2.z
        yaw = 0 # will adjust in the validation

        # Create Transform
        trigger_transform = carla.Transform(carla.Location(x, y, z), carla.Rotation(yaw))

        # Validate
        valid_trigger_transform = is_scenario_at_route(trigger_transform, route)
        if valid_trigger_transform:
            generated_trigger_points.append(valid_trigger_transform)

        attempts += 1

    if len(generated_trigger_points) < num_points:
        print(f"Only generated {len(generated_trigger_points)} valid points out of requested {num_points} after {max_attempts} attempts.")

    return generated_trigger_points

# def mutate_trigger_points(trigger_points, route, noise_level=0.3, max_attempts=100):
    
#     mutated_trigger_points = []

#     for tp in trigger_points:
#         valid = False
#         attempts = 0

#         while not valid and attempts < max_attempts:
#             # Add noise to location
#             x = add_noise(tp.location.x, noise_level)
#             y = add_noise(tp.location.y, noise_level)
#             z = tp.location.z

#             # Add noise to yaw
#             new_yaw = add_noise(tp.rotation.yaw, noise_level)

#             # Create mutated transform
#             mutated = carla.Transform(carla.Location(x, y, z), carla.Rotation(new_yaw))

#             if is_scenario_at_route(mutated, route):
#                 mutated_trigger_points.append(mutated)
#                 valid = True
#                 break

#             attempts += 1

#         if not valid:
#             print(f"Warning: Could not validate mutated point after {max_attempts} attempts. Skipping point: {tp.location}")

#     return mutated_trigger_points

def mutate_trigger_points(trigger_points, route, noise_level=0.3, max_attempts=100):
    """
    Mutate trigger points by adding noise along the direction of the yaw angle.
    """
    mutated_trigger_points = []

    for tp in trigger_points:
        valid = False
        attempts = 0

        while not valid and attempts < max_attempts:
            # Add noise to yaw
            new_yaw = add_noise(tp.rotation.yaw, noise_level)

            # Compute direction vector from yaw (convert to radians)
            yaw_rad = math.radians(new_yaw)
            dx = math.cos(yaw_rad)
            dy = math.sin(yaw_rad)

            # Add noise along heading direction
            magnitude = random.uniform(-noise_level, noise_level)
            x = tp.location.x + magnitude * dx
            y = tp.location.y + magnitude * dy
            z = tp.location.z  # keep z the same

            # Create mutated transform
            mutated = carla.Transform(carla.Location(x, y, z), carla.Rotation(new_yaw))

            if is_scenario_at_route(mutated, route):
                mutated_trigger_points.append(mutated)
                valid = True
                break

            attempts += 1

        if not valid:
            print(f"Warning: Could not validate mutated point after {max_attempts} attempts. Skipping point: {tp.location}")

    return mutated_trigger_points

def write_trigger_points_to_xml(original_xml_path, output_xml_path, mutated_trigger_points):

    tree = ET.parse(original_xml_path)
    root = tree.getroot()
    
    # Find all trigger_point elements in order
    all_trigger_elements = root.findall('.//trigger_point')
    
    if len(all_trigger_elements) != len(mutated_trigger_points):
        raise ValueError(f"Mismatch: {len(all_trigger_elements)} trigger points in XML vs. {len(mutated_trigger_points)} mutated points.")

    # Replace the trigger point attributes
    for elem, mutated_tp in zip(all_trigger_elements, mutated_trigger_points):
        elem.set('x', str(round(mutated_tp.location.x, 2)))
        elem.set('y', str(round(mutated_tp.location.y, 2)))
        elem.set('z', str(round(mutated_tp.location.z, 2)))
        elem.set('yaw', str(round(mutated_tp.rotation.yaw, 2)))

    # Save the modified XML
    tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)
    print(f"XML with updated trigger points saved as {output_xml_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add small random noise to positions in XML route file')
    parser.add_argument('--host', default='127.0.0.1', help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default='2000', help='TCP port to listen to (default: 2000)')
    parser.add_argument('--input_file', help='Input XML file', default='srunner/data/demo.xml',)
    parser.add_argument('--output_file', help='Output XML file', default='output.xml')
    parser.add_argument('--visualization_file', help='Visualization file', default='mutation_visualization.png')
    parser.add_argument('--visualize', action="store_true", help='Whether to visualize', default='mutation_visualization.png')
    parser.add_argument('--mutate', action="store_true", help='Mutate exising trigger points from configuration', default=True)
    parser.add_argument('--generate', action="store_true", help='Generate new trigger points', default=False)
    parser.add_argument('--noise_level', type=float, help='Noise level (default: 10)', default=10)
    
    args = parser.parse_args()
    
    if args.mutate or args.generate:
        client_timeout = 10.0
        client = carla.Client(args.host, int(args.port))
        client.set_timeout(client_timeout)
        CarlaDataProvider.set_client(client)

        route_configurations = RouteParser.parse_routes_file(route_filename=args.input_file)
        for config in route_configurations:
            world = client.load_world(config.town)
            world = client.get_world()
            CarlaDataProvider.set_world(world)
            route = get_route(config)
            config.scenario_configs
            
            if args.mutate:
                org_trigger_points = [scenario_config.trigger_points[0] for scenario_config in config.scenario_configs]
                trigger_points = mutate_trigger_points(org_trigger_points, route, noise_level=args.noise_level, max_attempts=100)
                for org_tp, mut_tp in zip(org_trigger_points,trigger_points):
                    print(f"Original trigger point: {org_tp.location}, yaw: {org_tp.rotation.yaw}")
                    print(f"Mutated trigger point: {mut_tp.location}, yaw: {mut_tp.rotation.yaw}")
            elif args.generate:
                trigger_points = add_random_trigger_points_along_path(config.keypoints, route, num_points=5, max_attempts=1000)
                for tp in trigger_points:
                    print(f"Generated trigger point: {tp.location}, yaw: {tp.rotation.yaw}")
            
            write_trigger_points_to_xml(args.input_file, args.output_file, trigger_points)

        world = client.load_world(config.town)
        world = client.get_world()
        ego_transforms, bg_transforms, bg_speeds = extract_scenario_inputs(args.input_file, world, bg_speed=2.0)
        # visualize_trajectories(world, ego_transforms, bg_transforms, bg_speeds, mode='straight')
        
    if args.visualize:
        all_collision_rate = visualize_combined(
            args.input_file, world, ego_transforms, bg_transforms, bg_speeds,
            save_path=args.visualization_file, mode='straight',
        )



