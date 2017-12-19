#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import os
import unittest

try:
    from setuptools import setup, find_packages, Command
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, find_packages, Command

import sphinxtogithub
from sphinxtogithub.tests import (
    filehandler,
    directoryhandler,
    replacer,
    renamer,
    remover,
    layout,
    layoutfactory,
    setup as setuptest,
)

class RunTests(Command):
    description = "Run the sphinxtogithub test suite."

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        suites = [
            filehandler.testSuite(),
            directoryhandler.testSuite(),
            replacer.testSuite(),
            renamer.testSuite(),
            remover.testSuite(),
            layout.testSuite(),
            layoutfactory.testSuite(),
            setuptest.testSuite(),
        ]

        suite = unittest.TestSuite(suites)

        runner = unittest.TextTestRunner()

        runner.run(suite)


class Publish(Command):
    description = "Publish package to PyPi"

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Publish to PyPi"""

        os.system("python setup.py sdist upload")


long_description = codecs.open("README.rst", "r", "utf-8").read()

setup(
    name='sphinxtogithub',
    version=sphinxtogithub.__version__,
    description=sphinxtogithub.__doc__,
    author=sphinxtogithub.__author__,
    author_email=sphinxtogithub.__contact__,
    url=sphinxtogithub.__homepage__,
    platforms=["any"],
    license="BSD",
    packages=find_packages(),
    scripts=["bin/sphinxtogithub"],
    zip_safe=False,
    install_requires=[],
    cmdclass = {"test": RunTests, "publish" : Publish},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Environment :: Plugins",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX",
        "Topic :: Documentation",
    ],
    long_description=long_description,
)
