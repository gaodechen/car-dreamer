import xml.etree.ElementTree as ET
import carla
from mc_simulation import run_monte_carlo_simulation, visualize_trajectories


def get_monte_carlo_score(world, route_file, route_id=None, debug=True):
    """
    Loads a route configuration from an XML file and runs a Monte Carlo simulation to compute a collision score.

    Args:
        world (carla.World): The CARLA world instance.
        route_file (str): Path to the XML configuration file.
        route_id (str, optional): Specific route ID to use. Defaults to the first route found.
        debug (bool, optional): If True, visualize trajectories after simulation.

    Returns:
        float: The collision rate from the Monte Carlo simulation.
    """
    # Parse the XML configuration file.
    tree = ET.parse(route_file)
    root = tree.getroot()

    # Find the route element (either by ID or take the first one).
    route = None
    if route_id:
        for r in root.findall('route'):
            if r.get('id') == route_id:
                route = r
                break
    else:
        route = root.find('route')

    if route is None:
        raise ValueError("No route found in the configuration file.")

    # Extract the first waypoint for the ego vehicle.
    waypoints = route.find('waypoints')
    if waypoints is None or len(waypoints) == 0:
        raise ValueError("No waypoints found for the ego vehicle.")

    first_wp = waypoints[0]
    ego_location = carla.Location(
        x=float(first_wp.get('x')),
        y=float(first_wp.get('y')),
        z=float(first_wp.get('z'))
    )
    # Assuming default yaw (rotation) of 0 if not provided.
    ego_transform = carla.Transform(ego_location, carla.Rotation(yaw=0.0))

    # Extract background vehicle positions from scenario trigger points.
    bg_transforms = []
    bg_speeds = []
    default_speed = 10.0  # default speed in m/s

    scenarios = route.find('scenarios')
    if scenarios is not None:
        for scenario in scenarios.findall('scenario'):
            trigger = scenario.find('trigger_point')
            if trigger is not None:
                loc = carla.Location(
                    x=float(trigger.get('x')),
                    y=float(trigger.get('y')),
                    z=float(trigger.get('z'))
                )
                yaw = float(trigger.get('yaw')) if 'yaw' in trigger.attrib else 0.0
                bg_transforms.append(carla.Transform(loc, carla.Rotation(yaw=yaw)))
                bg_speeds.append(default_speed)

    if not bg_transforms:
        raise ValueError("No background vehicles found in the configuration file.")

    # Run the Monte Carlo simulation.
    collision_rate, trajectories = run_monte_carlo_simulation(world, ego_transform, bg_transforms, bg_speeds)

    if debug:
        visualize_trajectories(world, ego_transform, bg_transforms, bg_speeds)

    return collision_rate

# Example usage:
# Assuming you already have a CARLA world instance:
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)
world = client.get_world()

#route_file = 'output.xml'
route_file = 'routes_town10.xml'
score = get_monte_carlo_score(world, route_file, debug=True)
print("Collision rate:", score)