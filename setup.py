import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="timepp",
    version="0.1.0",
    author="Thibaut Horel",
    author_email="thibauth@mit.edu",
    description="Simulation of temporal point processes",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    py_modules = ["timepp"],
    install_requires = [
        'scipy',
        ],
    url="https://github.com/Thibauth/timepp",
    python_requires='>=3.6',
)
