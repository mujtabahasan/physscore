from setuptools import setup, find_packages

setup(
    name="physscore",
    version="0.2.0",
    description="Physics-based plausibility metrics for 3D human pose estimation",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
    ],
)
