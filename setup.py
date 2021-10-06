import numpy 
from setuptools import setup
from Cython.Build import cythonize

#extensions = [
#    Extension("quadMeshProcessing", 
#    sources=["quadMeshProcessing.pyx"],
#    include_dirs=[numpy.get_include()], 
#    #extra_compile_args=["-O3"], 
#    language="c++")
#]

setup(
    ext_modules=cythonize("quadMeshProcessing.pyx", annotate=True,language_level=3),
    include_dirs=[numpy.get_include()]
)
