from setuptools import setup, find_packages

from primerize.__init__ import __version__

setup(
    name="primerize",
    description="PCR Assembly Primer Design",
    keywords="primerize PCR assembly misprime",
    version=__version__,
    author="Siqi Tian, Rhiju Das",
    author_email="rhiju@stanford.edu",
    url="https://github.com/ribokit/Primerize/",
    license="MIT",
    packages=find_packages(),
    install_requires=open("pyproject.toml", "r").readlines(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.10",
    zip_safe=True,
)
