"""
Setup configuration for DOE-Toolkit.
"""

from setuptools import setup, find_packages

setup(
    name="doe-toolkit",
    version="0.1.0",
    description="Free, open-source Design of Experiments software",
    author="DOE-Toolkit Contributors",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "scikit-learn",
        "cvxpy",
        "matplotlib",
        "plotly",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
        ],
    },
)
