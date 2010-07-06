from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import os

include_dirs = [numpy.get_include()]
if 'GSL_INCLUDE' in os.environ:
    include_dirs.append(os.environ['GSL_INCLUDE'])

ext_modules = [
    Extension("spline", ["spline.pyx"],
              libraries=['gsl', 'gslcblas'],
              include_dirs=include_dirs)]

# If one has e.g. ATLAS installed, one can link with it using:
#
#    libraries=['gsl', 'cblas', 'blas', 'atlas']
#
# to improve performance

setup(
  name = 'GSL splines',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
