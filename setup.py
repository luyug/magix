from setuptools import setup, find_packages, find_namespace_packages

setup(
    name='magix',
    version='0.0.1',
    packages=find_namespace_packages(include=['magix*']),
    license='Apache 2.0',
    author='Luyu Gao',
    author_email='luyug@cs.cmu.edu',
    python_requires='>=3.10',
    install_requires=[
        'transformers>=4.10.0',
        'datasets>=1.1.3',
        'simple_parsing',
        'sentencepiece',
    ]
)
