import os
from setuptools import find_packages, setup


here = os.path.abspath(os.path.dirname(__file__))

# Avoids IDE errors, but actual version is read from version.py
__version__ = None
with open("melior_transformers/version.py") as f:
    exec(f.read())

# Get the long description from the README file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="melior_transformers",
    version=__version__,
    author="MeliorAI",
    author_email="flavius@melior.ai",
    description="An easy-to-use wrapper library for the Transformers library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MeliorAI/meliorTransformers/",
    packages=find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch==1.3.1",
        "tqdm==4.41.1",
        "transformers==2.3.0",
        "numpy==1.18.1",
        "pandas==0.25.3",
        "seqeval==0.0.12",
        "scipy==1.4.1",
        "apex==0.9.10dev",
        "scikit_learn==0.22.1",
        "tensorboardX==2.0",
        "wandb==0.8.21",
        "requests",
        "regex",
        "wandb",
        "coloredlogs",
        "sentence-transformers==0.2.5",
    ],
)
