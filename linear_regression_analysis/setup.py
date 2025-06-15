from setuptools import setup

package_name = 'linear_regression_analysis'

setup(
    name=package_name,
    version='1.0.0',
    packages=['linear_regression_analysis'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/analysis_launch.py']),
        ('share/' + package_name + '/data', [
            'data/new_height_weight.csv',
            'data/HumanBrain_WeightandHead_size.csv',
            'data/boston_housing.csv'
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Berke',
    maintainer_email='berke.cevik@ue-germany.de',
    description='Linear Regression Multi Dataset Analysis for ROS2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'height_weight_node = linear_regression_analysis.height_weight_node:main',
            'brain_weight_node = linear_regression_analysis.brain_weight_node:main',
            'boston_housing_node = linear_regression_analysis.boston_housing_node:main',
            'analysis_launcher = linear_regression_analysis.analysis_launcher:main',
        ],
    },
)
