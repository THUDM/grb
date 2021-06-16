from codecs import open
from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    readme = fh.read()

setup(
    name='grb',
    version='0.0.1',
    description='Graph Robustness Benchmark',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Qinkai Zheng',
    author_email='qinkai.zheng1028@gmail.com',
    url='https://github.com/THUDM/grb',
    license="MIT",
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "scipy",
        "matplotlib",
        "torch",
        "numpy",
        "networkx",
        "recommonmark",
        "pandas",
        "cogdl",
        "scikit-learn",
    ],
    packages=find_packages(),
)
