# setup.py
from setuptools import setup, find_packages
import pathlib

HERE = pathlib.Path(__file__).parent
README_PATH = HERE / "README.md"

setup(
    name="aiwaf",
    version="0.1.9.7.6",
    description="AI-driven, self-learning Web Application Firewall for Python web applications",
    long_description=README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else "AIWAF",
    long_description_content_type="text/markdown",
    author="Aayush Gauba",
    url="https://github.com/aayushgauba/aiwaf",
    license="MIT",
    packages=find_packages(exclude=["tests*", "docs*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=2.4.6",
        "pandas>=3.0.3",
        "scikit-learn>=1.9.0,<2.0",
        "joblib>=1.5.3",
        "geoip2>=5.2.0",
        "packaging>=26.2",
        "requests>=2.34.2",
        "python-whois>=0.9.6",
    ],
    extras_require={
        "django": [
            "Django>=6.0.6",
        ],
        "flask": [
            "Flask>=3.1.3",
            "Flask-SQLAlchemy>=3.1.1",
        ],
        "fastapi": [
            "fastapi>=0.138.0",
            "starlette>=1.3.1",
            "uvicorn>=0.49.0",
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
