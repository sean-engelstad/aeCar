"""
Setup file for aecar package
run "pip install -e ." in this dir to install package
"""

from setuptools import setup

setup(
    name="aecar",
    version="1.0",
    author="Sean Engelstad",
    author_email="sengelstad312@gatech.edu",
    description="Online Reinforcement Learning Methods for Autonomous Vehicles",
    packages=[
        "aecar"
        ],
    package_dir = {'': 'src'},
    python_requires=">=3.6",
)