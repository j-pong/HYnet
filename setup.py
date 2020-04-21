import os
from setuptools import find_packages
from setuptools import setup

requirements = {
    "install": [
        "setuptools>=38.5.1",
        "configargparse>=1.2.1",
        "typeguard>=2.7.0",
        "dataclasses",  # For Python<3.7
        "humanfriendly",
        "scipy>=1.4.1",
        "matplotlib>=3.1.0",
        "seaborn",
        "pillow>=6.1.0",
        "editdistance==0.5.2",
        "tqdm",
        # Signal processing related
        "librosa>=0.7.0",
        "resampy",
        "pysptk>=0.1.17",
        # Natural language processing related
        "sentencepiece>=0.1.82",
        "nltk>=3.4.5",
        # File IO related
        "PyYAML>=5.1.2",
        "soundfile>=0.10.2",
        "h5py==2.9.0",
        "kaldiio>=2.15.0",
    ],
}

install_requires = requirements["install"]

dirname = os.path.dirname(__file__)
setup(name='moneynet',
      version='0.0.3',
      author='jpong',
      author_email='ljh93ljh@gmail.com',
      description='moneynet',
      license='Apache Software License',
      packages=find_packages(include=['moneynet*']),
      install_requires=install_requires,
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