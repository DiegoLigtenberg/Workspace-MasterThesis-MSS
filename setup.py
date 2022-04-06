from setuptools import setup, find_packages
setup(
    name="mss_project", 
    version="1.0", 
    author="Diego Ligtenberg",
    author_email="diegoligtenberg@gmail.com",
    description="Master Thesis about Music Source Separation, and Instrument classification",
    license="MIT",
    long_description=("README"),
    packages=find_packages()
    )

# print(find_packages())
# can install this with:     pip install -e .
# can uninstall this with:   pip uninstall mss_project
