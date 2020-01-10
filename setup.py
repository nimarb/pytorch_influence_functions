import setuptools
from pathlib import Path

this_dir = Path(__file__).resolve().parent
with open(this_dir / 'README.md', "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch_influence_functions",
    version="0.1.1",
    author="Nimar Blume",
    author_email="l.nimar.b@gmail.com",
    description="This package is a plug-n-play PyTorch reimplementation of Influence Functions. Influence Functions were introduced in the paper Understanding Black-box Predictions via Influence Functions by Pang Wei Koh and Percy Liang (ICML2017).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nimarb/pytorch_influence_functions",
    packages=setuptools.find_packages(),
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6',
    install_requires=[
          'torch>=1.0',
          'numpy>=1.13.0'
    ]
)
