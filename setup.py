# from distutils.core import setup
from setuptools import setup

setup(
    name='therpy',
    version='0.3.4',
    packages=['therpy',],

    # Package requirements
    install_requires=['tqdm', 'astropy'],

    # metadata
    author='Parth Patel',
    author_email='ranchop09@gmail.com',
    license='Open Source',
)

# Development Mode
# python setup.py develop
# python setup.py develop --uninstall

# Deploying the Package Locally
# python setup.py install

# Deploying the Package on PyPI
# python setup.py sdist
# python setup.py register (ONLY FIRST TIME)
# python setup.py sdist upload

# More information available at
# http://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode

# Complete Example
# from setuptools import setup, find_packages
# setup(
#     name="HelloWorld",
#     version="0.1",
#     packages=find_packages(),
#     scripts=['say_hello.py'],
#
#     # Project uses reStructuredText, so ensure that the docutils get
#     # installed or upgraded on the target machine
#     install_requires=['docutils>=0.3'],
#
#     package_data={
#         # If any package contains *.txt or *.rst files, include them:
#         '': ['*.txt', '*.rst'],
#         # And include any *.msg files found in the 'hello' package, too:
#         'hello': ['*.msg'],
#     },
#
#     # metadata for upload to PyPI
#     author="Me",
#     author_email="me@example.com",
#     description="This is an Example Package",
#     license="PSF",
#     keywords="hello world example examples",
#     url="http://example.com/HelloWorld/",   # project home page, if any
#
#     # could also include long_description, download_url, classifiers, etc.
# )
