"""Build script for Cython-accelerated game engine."""
from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "jaipur._engine",
        ["jaipur/_engine.pyx"],
    ),
]

setup(
    packages=["jaipur"],
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
            "nonecheck": False,
        },
    ),
)
