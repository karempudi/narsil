from setuptools import setup

setup(
        name='Narsil',
        version='0.1.0',
        author='Praneeth Karempudi',
        author_email='praneeth.karempudi@gmail.com',
        license='LICENSE.txt',
        description='The package to work with microscopy data of prokaryotic cells',
        long_description=open('README.md').read(),
        install_requires=[
            'numpy',
            'torch',
            'torchvision',
            'scikit-image',
            'scipy',
            'matplotlib'
            ]
        )
