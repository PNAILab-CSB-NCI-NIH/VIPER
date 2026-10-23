from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).resolve().parent

requirements = [
    line.strip()
    for line in (ROOT / "requirements.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]

setup(
    name="VIPER",
    version="1.0.0",
    author="Hermann F. Degenhardt, Erin Lee, Maximilia F. S. Degenhardt",
    description="Self-supervised AFM denoising with early-stopping regularization",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/PNAILab-CSB-NCI-NIH/VIPER",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
)