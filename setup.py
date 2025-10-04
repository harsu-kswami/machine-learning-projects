#!/usr/bin/env python3
"""Setup script for Positional Encoding Visualizer."""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="positional-encoding-visualizer",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Interactive educational tool for understanding transformer positional encodings",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/positional-encoding-visualizer",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/positional-encoding-visualizer/issues",
        "Documentation": "https://github.com/yourusername/positional-encoding-visualizer/docs",
        "Source Code": "https://github.com/yourusername/positional-encoding-visualizer",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.971",
            "pre-commit>=2.20.0",
        ],
        "gpu": [
            "torch-audio>=0.12.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.971",
            "pre-commit>=2.20.0",
            "torch-audio>=0.12.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "pos-vis-dashboard=src.interactive.streamlit_app:main",
            "pos-vis-benchmark=scripts.benchmark_encodings:main",
            "pos-vis-export=scripts.export_visualizations:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.json", "*.yaml", "*.yml"],
        "data": ["*.txt", "*.json"],
        "assets": ["images/*", "animations/*"],
    },
    zip_safe=False,
    keywords=[
        "transformer",
        "attention",
        "positional-encoding",
        "visualization",
        "pytorch",
        "machine-learning",
        "deep-learning",
        "nlp",
        "educational",
        "interactive",
    ],
)
