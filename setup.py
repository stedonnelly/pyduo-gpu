"""
PyDuo - GPU-accelerated diatomic rovibronic calculations

A Python/CUDA implementation of the Duo program.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyduo",
    version="0.1.0",
    author="PyDuo Contributors",
    author_email="",
    description="GPU-accelerated diatomic rovibronic calculations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ExoMol/Duo",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "gpu": ["cupy>=12.0.0"],
        "full": ["cupy>=12.0.0", "scipy>=1.7.0"],
        "dev": ["pytest>=7.0.0", "black>=22.0.0", "mypy>=0.990"],
    },
    entry_points={
        "console_scripts": [
            "pyduo=pyduo.duo:main",
        ],
    },
)
