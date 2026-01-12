#!/usr/bin/env python3
"""
Setup script for Movie Recommendation System.
"""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md") as f:
    long_description = f.read()

setup(
    name="movie-recommendation-system",
    version="1.0.0",
    description="A comprehensive movie recommendation system with multiple algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Movie Recommendation System Team",
    author_email="team@example.com",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "movie-recommendation=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
