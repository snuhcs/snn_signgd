from setuptools import setup, find_packages
import os

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='snn_signgd',
    version='0.1.0',
    description='Optimization based SNN',
    author='Hyunseok Oh',
    author_email='ohsai@snu.ac.kr',
    packages=find_packages(where=os.path.join('src', 'packages')),  # Specify only the 'logical_z3' package
    package_dir={'': os.path.join('src', 'packages')},  # Map the package to the 'src/logical_z3' directory
    install_requires=install_requires,
    keywords=[
        'artificial intelligence',
        'deep learning',
        'optimization theory',
        'spiking neural network',
        'ann-to-snn conversion',
        'neuronal dynamics',
        'sign gradient descent',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.10',
    ],
)
