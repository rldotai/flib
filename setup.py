#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup, Command
except ImportError:
    from distutils.core import setup, Command


class PyTest(Command):
    user_options = []
    def initialize_options(self):
        pass

    def finalize_options(self):
        self.test_args = []
        self.test_suite = True

    def run(self):
        import sys
        import pytest
        sys.exit(pytest.main(self.test_args))


with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

requirements = []

test_requirements = [
    'flake8',
    'pytest==2.6.4',
    'pytest-cov==1.8',
]

setup(
    name='flib',
    version='0.0.0',
    description="Modular feature vector library for machine learning",
    long_description=readme + '\n\n' + history,
    author="bab",
    author_email='rldot41@gmail.com',
    url='https://github.com/rldotai/flib',
    packages=[
        'flib',
    ],
    package_dir={'flib':
                 'flib'},
    include_package_data=True,
    install_requires=requirements,
    license="BSD",
    zip_safe=False,
    keywords='flib',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    cmdclass={'test': PyTest},
    tests_require=test_requirements
)
