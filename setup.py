from setuptools import find_packages
from setuptools import setup

install_requires = [
    'numpy',
    'scipy',
    'autorom[accept-rom-license]',
    'gymnasium[accept-rom-license,atari]',
    'pfrl@git+https://github.com/prabhatnagarajan/pfrl@ewrl_experiments',
    'matplotlib',
    'typing_extensions>=4.5.0',
    'minatar',
    'ale_utils@git+https://github.com/prabhatnagarajan/ale_utils@main',
    'plot_pfrl@git+https://github.com/prabhatnagarajan/plot_pfrl.git@main',
    'opencv-python',
    'arch==8.0.0',
    'table-rl',
    'swig',
    'gymnasium[box2d]', 
    ]

test_requires = [
    "pytest",
    "attrs<19.2.0",  # pytest does not run with attrs==19.2.0 (https://github.com/pytest-dev/pytest/issues/3280)  # NOQA
]

setup(
    name="me_layer",
    version="1.0.0",
    description="Official Repository for ICML 2026 Paper: Accelerating Q-learning through Efficient Value-Sharing across Actions",
    keywords="mean-expansion layer, ME layer, dueling, DQN",
    author="Prabhat Nagarajan",
    author_email="nagarajan@ualberta.ca",
    packages=find_packages(),
    install_requires=install_requires,
    test_requires=test_requires,
)
