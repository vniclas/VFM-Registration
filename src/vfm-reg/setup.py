from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['vfm_reg'],
    package_dir={'': 'src'},
)

setup(**setup_args)
