from setuptools import setup, find_packages
# from codecs import open
# from os import path

__author__ = 'Salvatore Citraro'
__license__ = "BSD-2-Clause"
__email__ = "salvatore.citraro@phd.unipi.it"

# here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
# with open(path.join(here, 'README.md'), encoding='utf-8') as f:
#    long_description = f.read()


setup(name='xmark',
      version='1.0.0',
      license='BSD-Clause-2',
      description='XMark: Benchmark for Node-Attributed Community Discovery',
      url='https://github.com/dsalvaz/XMark',
      author=['Salvatore Citraro'],
      author_email='salvatore.citraro@phd.unipi.it',
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 5 - Production/Stable',

          # Indicate who your project is intended for
          'Intended Audience :: Developers',
          'Topic :: Software Development :: Build Tools',

          # Pick your license as you wish (should match "license" above)
          'License :: OSI Approved :: BSD License',

          "Operating System :: OS Independent",

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          'Programming Language :: Python :: 3'
      ],
      keywords='complex-networks community-discovery labeled-graph network-generator',
      install_requires=['numpy', 'networkx', ''],
      packages=find_packages(exclude=["*.test", "*.test.*", "test.*", "test", "xmark.test", "xmark.test.*"]),
      )
