import os
from setuptools import find_packages
from setuptools import setup

dirname = os.path.dirname(__file__)
setup(name='moneynet',
      version='0.0.3',
      author='jpong',
      author_email='ljh93ljh@gmail.com',
      description='moneynet',
      license='Apache Software License',
      packages=find_packages(include=['moneynet*']),
      classifiers=[
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Topic :: Software Development :: Libraries :: Python Modules'],
      )