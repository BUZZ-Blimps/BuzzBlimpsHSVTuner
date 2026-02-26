from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'blimp_vision'

# Function to get all files recursively from a directory
def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Add launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Add config files if you have any
        (os.path.join('share', package_name, 'calibration'), glob('calibration/*.yaml')),
        # Add model files with recursive directory structure
        *[
            (os.path.join('share', package_name, os.path.dirname(p)), [p])
            for p in glob('models/**/*', recursive=True)
            if os.path.isfile(p)
        ],
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='sah4j.patel@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'blimp_vision_node = blimp_vision.camera_node:main',
        ],
    },
)
