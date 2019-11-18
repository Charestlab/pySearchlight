from setuptools import setup, find_packages

requires = []
with open('requirements.txt') as reqfile:
    requires = reqfile.read().splitlines()

with open('README.md', encoding='utf-8') as readmefile:
    long_description = readmefile.read()


setup(
    name='pySearchlight',
    version='0.0.1',
    description='pySearchlight',
    url='https://github.com/charestlab/pySearchlight',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 1 - Planning",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
    ],
    maintainer='Ian Charest, Daniel Lindh and Jasper van den Bosch',
    maintainer_email='i.charest@bham.ac.uk',
    keywords='neuroscience ',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=requires,
    tests_require=requires,
    test_suite='tests'   
)
