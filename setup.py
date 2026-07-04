from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'track_drive'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # launch 파일을 설치
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'traffic_light = track_drive.traffic:main',
            'human = track_drive.human:main',
            'lane_yurim = track_drive.lane_yurim:main',
            'traffic_jm = track_drive.traffic_jm:main',
            'state = track_drive.track_drive:main',
            'corn_jh = track_drive.corn_jh:main',
            'pp_jm = track_drive.pp_jm:main',
            'humen_detect= track_drive.humen_detect:main',
            'lidar_jh = track_drive.lidar:main',
            'lidar_pass = track_drive.lidar_pass:main',
            'side_camera_viewer = track_drive.side_camera_viewer:main',
        ],
    },
)
