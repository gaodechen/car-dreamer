#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the key configuration parameters for an XML-based scenario
"""

import carla


class ActorConfigurationData(object):

    """
    This is a configuration base class to hold model and transform attributes
    """

    def __init__(self, model, transform, rolename='other', speed=0, autopilot=False,
                 random=False, color=None, category="car", args=None):
        self.model = model
        self.rolename = rolename
        self.transform = transform
        self.speed = speed
        self.autopilot = autopilot
        self.random_location = random
        self.color = color
        self.category = category
        self.args = args

    @staticmethod
    def parse_from_node(node, rolename):
        """
        static method to initialize an ActorConfigurationData from a given ET tree
        """

        model = node.attrib.get('model', 'vehicle.*')

        pos_x = float(node.attrib.get('x', 0))
        pos_y = float(node.attrib.get('y', 0))
        pos_z = float(node.attrib.get('z', 0))
        yaw = float(node.attrib.get('yaw', 0))

        transform = carla.Transform(carla.Location(x=pos_x, y=pos_y, z=pos_z), carla.Rotation(yaw=yaw))

        rolename = node.attrib.get('rolename', rolename)

        speed = node.attrib.get('speed', 0)

        autopilot = False
        if 'autopilot' in node.keys():
            autopilot = True

        random_location = False
        if 'random_location' in node.keys():
            random_location = True

        color = node.attrib.get('color', None)

        return ActorConfigurationData(model, transform, rolename, speed, autopilot, random_location, color)

    @staticmethod
    def parse_from_dict(actor_dict, rolename):
        """
        static method to initialize an ActorConfigurationData from a given ET tree
        """

        model = actor_dict['model'] if 'model' in actor_dict else 'vehicle.*'

        pos_x = float(actor_dict['x']) if 'x' in actor_dict else 0
        pos_y = float(actor_dict['y']) if 'y' in actor_dict else 0
        pos_z = float(actor_dict['z']) if 'z' in actor_dict else 0
        yaw = float(actor_dict['yaw']) if 'yaw' in actor_dict else 0
        transform = carla.Transform(carla.Location(x=pos_x, y=pos_y, z=pos_z), carla.Rotation(yaw=yaw))

        rolename = actor_dict['rolename'] if 'rolename' in actor_dict else rolename
        speed = actor_dict['speed'] if 'speed' in actor_dict else 0
        autopilot = actor_dict['autopilot'] if 'autopilot' in actor_dict else False
        random_location = actor_dict['random_location'] if 'random_location' in actor_dict else False
        color = actor_dict['color'] if 'color' in actor_dict else None

        return ActorConfigurationData(model, transform, rolename, speed, autopilot, random_location, color)


class ScenarioConfiguration(object):

    """
    This class provides a basic scenario configuration incl.:
    - configurations for all actors
    - town, where the scenario should be executed
    - name of the scenario (e.g. ControlLoss_1)
    - type is the class of scenario (e.g. ControlLoss)
    """

    def __init__(self):
        self.trigger_points = []
        self.ego_vehicles = []
        self.other_actors = []
        self.other_parameters = {}
        self.town = None
        self.name = None
        self.type = None
        self.route = None
        self.agent = None
        self.weather = carla.WeatherParameters(sun_altitude_angle=70, cloudiness=50)
        self.friction = None
        self.subtype = None
        self.route_var_name = None

    def print_all(self):
        print("trigger_points:", self.trigger_points[0].location)
        print("ego_vehicles model:", self.ego_vehicles[0].model)
        print("ego_vehicles transform:", self.ego_vehicles[0].transform)
        print("ego_vehicles speed:", self.ego_vehicles[0].speed)
        print("ego_vehicles random_location:", self.ego_vehicles[0].random_location)
        print("ego_vehicles args:", self.ego_vehicles[0].args)
        # print("other_actors:", self.other_actors)
        # print("other_parameters:", self.other_parameters)
        # print("town:", self.town)
        # print("name:", self.name)
        # print("type:", self.type)
        # print("route:", self.route)
        # print("agent:", self.agent)
        # print("weather:", self.weather)
        # print("friction:", self.friction)
        # print("subtype:", self.subtype)
        # print("route_var_name:", self.route_var_name)