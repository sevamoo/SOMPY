try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

setup(
    name="SOMPY",
    version="1.1",
    description="Self Organizing Maps Package",
    author="Vahid Moosavi and Sebastian Packmann",
    packages=find_packages(),
    install_requires=['numpy >= 1.7', 'scipy >= 0.9',
                      'scikit-learn >= 0.21', 'numexpr >= 2.5']
)
