#!/usr/bin/env python

"""Package configuration for pytorch-hccl-tests."""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
readme = (ROOT / "README.rst").read_text(encoding="utf-8")
history = (ROOT / "HISTORY.rst").read_text(encoding="utf-8")

requirements = [
    "Click>=7.0",
    "pandas==2.3.3",
    "numpy==1.26.4",
]

test_requirements = [
    "pytest>=8",
]

setup(
    author="HCCL Test Authors",
    author_email="hccl-tests@gmail.com",
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    description="End-to-end PyTorch distributed benchmarks (HCCL backend)",
    entry_points={
        "console_scripts": [
            "torch-hccl-benchs=pytorch_hccl_tests.cli:main",
        ],
    },
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="pytorch_hccl_tests",
    name="pytorch_hccl_tests",
    packages=find_packages(include=["pytorch_hccl_tests", "pytorch_hccl_tests.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/huawei-csl/pytorch-hccl-tests",
    version="0.1.14",
    zip_safe=False,
)
