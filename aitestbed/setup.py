"""
Setup script for the 6G AI Traffic Characterization Testbed.

This allows the package to be installed in development mode:
    pip install -e .

Or run directly:
    testbed --list-scenarios
"""

from setuptools import setup, find_packages

setup(
    name="ai-traffic-testbed",
    version="0.2.0",
    description="6G AI Traffic Characterization Testbed",
    author="3GPP SA4 6G Media Study",
    packages=find_packages(),
    py_modules=["orchestrator"],
    include_package_data=True,
    package_data={"configs": ["*.yaml", "*.json"]},
    python_requires=">=3.10",
    install_requires=[
        "openai>=1.0.0",
        "anthropic>=0.40.0",
        "azure-ai-inference>=1.0.0b1",
        "google-genai>=1.0.0",
        "websockets>=12.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyarrow>=14.0.0",
        "pyyaml>=6.0",
        "requests>=2.28.0",
        "python-dotenv>=1.0.0",
        "httpx>=0.28.0",
        "netemu[pcap]>=0.2.0",
        "dpkt>=1.9.8",
        "mcp>=1.0.0",
        "a2a-sdk>=1.1.0,<2.0.0",
        "uvicorn>=0.27.0",
        "starlette>=0.40.0",
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
