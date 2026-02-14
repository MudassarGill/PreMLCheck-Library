"""
PreMLCheck - Intelligent dataset analysis before ML training
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='premlcheck',
    version='0.1.0',
    author='Mudassar Hussain',
    author_email='mudassarhussain6533@gmail.com',
    description='An intelligent Python library that analyzes datasets before training machine learning models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/MudassarGill/PreMLCheck-Library',
    packages=find_packages(exclude=['tests', 'tests.*', 'examples', 'examples.*', 'docs']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.10',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0',
            'black>=21.0',
            'flake8>=3.9.0',
            'mypy>=0.900',
        ],
        'viz': [
            'matplotlib>=3.0.0',
            'seaborn>=0.11.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'premlcheck=premlcheck.cli:main',
        ],
    },
    include_package_data=True,
    keywords='machine-learning data-analysis preprocessing ml-advisor dataset-quality overfitting-detection',
    project_urls={
        'Bug Reports': 'https://github.com/MudassarGill/PreMLCheck-Library/issues',
        'Source': 'https://github.com/MudassarGill/PreMLCheck-Library',
        'Documentation': 'https://github.com/MudassarGill/PreMLCheck-Library/tree/main/docs',
    },
)
