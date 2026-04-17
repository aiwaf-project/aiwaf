# setup.py
from setuptools import setup, find_packages
import pathlib

HERE = pathlib.Path(__file__).parent
README_PATH = HERE / "README.md"

setup(
    name="aiwaf",
    version="0.1.9.7.2",
    description="AI-driven, self-learning Web Application Firewall for Python web applications",
    long_description=README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else "AIWAF",
    long_description_content_type="text/markdown",
    author="Aayush Gauba",
    url="https://github.com/aayushgauba/aiwaf",
    license="MIT",
    packages=find_packages(exclude=["tests*", "docs*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.3",
        "scikit-learn>=1.0,<2.0",
        "joblib>=1.1",
        "geoip2>=4.0",
        "packaging>=21.0",
        "requests>=2.25.0",
        "python-whois>=0.9.0",
    ],
    extras_require={
        "django": [
            "Django>=3.2",
        ],
        "flask": [
            "Flask>=2.3",
            "Flask-SQLAlchemy>=3.0",
        ],
        "fastapi": [
            "fastapi>=0.110",
            "starlette>=0.36",
            "uvicorn>=0.27",
        ],
        "rust": [
            "aiwaf-rust>=0.1.6",
        ],
    },
    include_package_data=True,
    package_data={
        # include your pretrained model and any JSON resources
        "aiwaf.django": ["resources/*.pkl", "resources/*.json"],
        "aiwaf.flask": ["resources/*.pkl", "resources/*.json"],
        "aiwaf.core": ["geolock/*.mmdb"],
    },
    entry_points={
        "console_scripts": [
            "aiwaf=aiwaf.cli:main",
            "aiwaf-detect=aiwaf.cli:aiwaf_detect",
            "aiwaf-fast=aiwaf.fast.cli:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Framework :: Django",
        "Framework :: Flask",
        "Framework :: FastAPI",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
    ],
)
