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
    version="0.18.2+melior1.1.0",,
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
        "numpy",
        "requests",
        "tqdm",
        "regex",
        "transformers",
        "scipy",
        "scikit-learn",
        "seqeval",
        "tensorboardx",
        "wandb"
    ],
)
