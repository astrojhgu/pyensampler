#!/usr/bin/env python3
from setuptools import setup
import ensampler

print("Setupping pyensampler version={0}".format(ensampler.__version__))
setup(
    name="ensampler",
    version=ensampler.__version__,
    packages=["ensampler"],
    zip_safe=True,
)
