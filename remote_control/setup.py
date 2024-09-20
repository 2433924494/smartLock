from setuptools import find_packages, setup

package_name = 'remote_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='LING',
    maintainer_email='2433924494@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "remote_control=remote_control.control:main",
            "test=remote_control.test_video:main",
            "video_send=remote_control.video_send:main",
            "command_subscriber=remote_control.command_subscriber:main",
        ],
    },
)
