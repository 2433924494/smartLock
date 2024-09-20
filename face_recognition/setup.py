from setuptools import find_packages, setup

package_name = 'face_recognition'

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
    maintainer='ling',
    maintainer_email='ling@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "node_test=face_recognition.node_test:main",
            "publisher_test=face_recognition.publisher_test:main",
            "subscriber_test=face_recognition.subscriber_test:main",
            "FaceRecognitionPublisher=face_recognition.FaceRecognitionPublisher:main",
            "test=face_recognition.node_test:main"
        ],
    },
)
