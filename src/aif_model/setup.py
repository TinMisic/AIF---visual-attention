from setuptools import find_packages, setup

package_name = 'aif_model'

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
    maintainer='tin',
    maintainer_email='misic.tin@gmail.com',
    description='Active inference model for visual attention',
    license='Apache2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'act_inf = aif_model.act_inf:main'
        ],
    },
)