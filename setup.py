from setuptools import setup, find_packages

setup(
        name='narsil',
        version='0.1.0',
        author='Praneeth Karempudi',
        author_email='praneeth.karempudi@gmail.com',
        license='LICENSE.txt',
        description='The package to work with microscopy data of prokaryotic cells',
        long_description=open('README.md').read(),
        url="https://github.com/karempudi/narsil",
        install_requires=[
            'numpy',
            'torch',
            'torchvision',
            'scikit-image',
            'scipy',
            'matplotlib',
            #'wxPython',
            'psycopg2',
            'PySide6',
            'pyqtgraph',
            'notebook',
            'pycromanager'
            ],
        packages=find_packages(exclude=("tests", "notebooks",))
        )
