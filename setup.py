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

upload_requires = [
    "twine==3.1.1",
]

install_requires = [
    "torch==1.3.1",
    "tqdm==4.41.1",
    "transformers==2.3.0",
    "numpy==1.16.3",
    "pandas==0.25.3",
    "seqeval==0.0.12",
    "scipy==1.2.1",
    # "apex==0.9.10dev",
    "scikit-learn~=0.20.2",
    "tensorboardX==2.0",
    "wandb==0.8.21",
    "requests",
    "regex",
    "wandb",
    "coloredlogs",
    "sentence-transformers==0.2.5",
]

tests_requires = [
    # test
    "pytest-cov==2.7.1",
    "pytest-localserver==0.5.0",
    "pytest==5.1.3",
    # lint/format/types
    "black==19.10b0",
    "flake8==3.7.8",
    "pytype==2019.7.11",
    "isort==4.3.21",
    "pre-commit==1.21.0",
]

extras_requires = {
    "test": tests_requires,
    "dev": tests_requires + upload_requires,
    "upload": upload_requires,
}

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
    install_requires=install_requires,
    tests_require=tests_requires,
    extras_require=extras_requires,
)
