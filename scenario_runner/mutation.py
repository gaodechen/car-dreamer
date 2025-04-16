import xml.etree.ElementTree as ET
import random
import copy
import argparse

def add_noise(value, noise_level=0.3):
    """Add small random noise to a numerical value."""
    noise = random.uniform(-noise_level, noise_level)
    return round(float(value) + noise, 1)

def process_xml(input_file, output_file, noise_level=0.3):
    # Parse the XML file
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    # Process waypoints positions
    for route in root.findall('./route'):
        for waypoints in route.findall('./waypoints'):
            for position in waypoints.findall('./position'):
                # Add noise to x, y, z attributes
                position.set('x', str(add_noise(position.get('x'), noise_level)))
                position.set('y', str(add_noise(position.get('y'), noise_level)))
                position.set('z', str(add_noise(position.get('z'), noise_level)))
        
        # Process scenario trigger points
        for scenarios in route.findall('./scenarios'):
            for scenario in scenarios.findall('./scenario'):
                for trigger_point in scenario.findall('./trigger_point'):
                    # Add noise to x, y, z attributes
                    trigger_point.set('x', str(add_noise(trigger_point.get('x'), noise_level)))
                    trigger_point.set('y', str(add_noise(trigger_point.get('y'), noise_level)))
                    trigger_point.set('z', str(add_noise(trigger_point.get('z'), noise_level)))
                    
                    # Slightly modify yaw if present
                    if 'yaw' in trigger_point.attrib:
                        trigger_point.set('yaw', str(add_noise(trigger_point.get('yaw'), noise_level)))
    
    # Write the modified XML to the output file
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"Modified XML saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add small random noise to positions in XML route file')
    parser.add_argument('--input_file', help='Input XML file', default='srunner/data/demo.xml',)
    parser.add_argument('--output_file', help='Output XML file (default: output.xml)', default='output.xml')
    parser.add_argument('--noise', type=float, help='Noise level (default: 0.5)', default=2)
    
    args = parser.parse_args()
    
    process_xml(args.input_file, args.output_file, args.noise)