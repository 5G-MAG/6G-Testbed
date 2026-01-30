"""
Setup script for the 6G AI Traffic Characterization Testbed.

This allows the package to be installed in development mode:
    pip install -e .

Or run directly:
    python -m testbed.orchestrator --list-scenarios
"""

from setuptools import setup, find_packages

setup(
    name="testbed",
    version="0.1.0",
    description="6G AI Traffic Characterization Testbed",
    author="3GPP SA4 6G Media Study",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "openai>=1.0.0",
        "google-generativeai>=0.3.0",
        "websockets>=12.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "requests>=2.28.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "mypy",
        ],
        "capture": [
            "mitmproxy>=10.0.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "webrtc": [
            "aiortc>=1.5.0",
        ],
        "computer-use": [
            "playwright>=1.41.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "testbed=orchestrator:main",
        ],
    },
)
