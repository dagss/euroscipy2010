# Tutorial Cython wrapper around GSL splines.
#
# Suggested reading order:
# 1. example1()
# 2. Spline fields, Spline.__cinit__, Spline.evaluate_single
# 3. resample
# 4. __reduce__
# 5. evaluate
#
# "Exercise for the reader":
# As coded here, Spline is only available as a Cython type in this module.
# By exporting Spline in spline.pxd, it can be made available to other
# modules, who can call evaluate_single for fast evaluation of single
# points.

import numpy as np # run-time NumPy scope
cimport numpy as np # compile-time NumPy scope
cimport cython
cimport errno
from python_unicode cimport PyUnicode_DecodeASCII

# Cython does not support "const", hackaround needed
from errno cimport const_char

cdef extern from "string.h":
    size_t strlen(const_char *s)

# In order to call NumPy C API functions such as np.PyArray_RemoveSmallest,
# we need to call this at module load time:
np.import_array()

# Install error handler for GSL. Forget about thread-safety in this
# example. This is a bit delicate because the memory data pointed to
# by reason and file may disappear once the function returns -- they
# must NOT simply be converted to strings, but copied!
current_gsl_exc = None

cdef raise_current_gsl_exc():
    global current_gsl_exc
    exc = current_gsl_exc
    current_gsl_exc = None
    raise exc

cdef void error_handler(const_char *reason,
                        const_char *filename,
                        int line,
                        int gsl_errno):
    global current_gsl_exc
    msg = "%s (GSL error %d, %s:%d)" % (
        PyUnicode_DecodeASCII(reason, strlen(reason), NULL),
        gsl_errno,
        PyUnicode_DecodeASCII(filename, strlen(filename), NULL),
        line)
    
    # TODO: Map error numbers to proper exception types
    current_gsl_exc = RuntimeError(msg)

errno.gsl_set_error_handler(&error_handler)

# Create a cdef class = Python C Extension Type cdef classes are
# allowed to store C types (e.g. pointers) in addition to Python
# objects.  They behave very similar to Python classes.
cdef class Spline:

    # Attributes in cdef classes must be declared
    # By default, they are not visible to Python-space, only
    # to Cython users of the class (which are using the class
    # through a typed variable)
    cdef gsl_spline *spline
    cdef gsl_interp_accel *acc
    # Attributes with Python objects can be made visible to Python
    # space via the keywords "public", "readonly", or by creating
    # property setters/getters (see Cython docs)
    cdef readonly np.ndarray x, y
    cdef readonly object algorithm # Really "str", but do not type unless necesarry!

    # Note the c in __cinit__!
    # cdef classes may have __init__ as well, but __init__ is not
    # guaranteed to be called. Since leaving self.spline and self.acc
    # uninitialized would be fatal (and crash the Python interpreter),
    # there's a __cinit__ which is guaranteed to be called.
    def __cinit__(self, x, y, algorithm='cubic'):
        cdef np.ndarray xarr, yarr
        cdef gsl_interp_type *interptype

        self.acc = NULL
        self.spline = NULL

        # GSL wants the x and y data as contiguous blocks of double
        # floating-point values, so make sure that is what we have.
        self.x = xarr = np.ascontiguousarray(x, dtype=np.double)
        self.y = yarr = np.ascontiguousarray(y, dtype=np.double)
        if yarr.shape[0] != xarr.shape[0] or not (xarr.ndim == yarr.ndim == 1):
            raise ValueError('x and y shapes do not match, or are not 1D')
        if algorithm == 'cubic':
            interptype = gsl_interp_cspline
        elif algorithm == 'akima':
            interptype = gsl_interp_akima
        else:
            raise NotImplementedError('Unimplemented spline type: %s' % algorithm)
        self.algorithm = algorithm
        
        self.spline = gsl_spline_alloc(interptype, xarr.shape[0])
        if self.spline == NULL:
            raise_current_gsl_exc()
        self.acc = gsl_interp_accel_alloc()
        if self.acc == NULL:
            gsl_spline_free(self.spline)
            self.spline = NULL
            raise_current_gsl_exc()
        if gsl_spline_init(self.spline, <double*>xarr.data, <double*>yarr.data,
                           xarr.shape[0]) != 0:
            raise_current_gsl_exc()

    # __dealloc__ should deallocate C memory that must be explicitly
    # freed
    def __dealloc__(self):
        if self.spline != NULL:
            gsl_spline_free(self.spline)
        if self.acc != NULL:
            gsl_interp_accel_free(self.acc)

    cdef double evaluate_single(self, double x):
        """
        Fast entry point for Cython callers, to look up a single
        scalar value. It avoids a lot of the overhead of evaluate.        
        """
        return gsl_spline_eval(self.spline, x, self.acc)

    def evaluate(self, x, out=None):
        """
        Evaluate the splined function for a set of values given in x.
        The result will be the same shape as x.
        """
        cdef gsl_interp_accel *acc = self.acc
        cdef gsl_spline *spline = self.spline

        if np.isscalar(x):
            return self.evaluate_single(x)
        
        x = np.asarray(x, dtype=np.double)
        if out is None:
            out = np.empty_like(x)

        # A simple way to iterate over many arrays simultaenously is
        # to use broadcast. Note that we don't really want to broadcast
        # here, since there's only one input; but this can readily be
        # generalized to many inputs which then broadcast properly.
        cdef np.broadcast mit = np.broadcast(x, out)
 
        if out.shape != mit.shape:
            raise ValueError("out array supplied does not match inputs")
        
        # Remove one of the dimensions from the multi-iterator, and
        # iterate over it manually using a (more efficient) loop
        cdef int innerdim = np.PyArray_RemoveSmallest(mit)
        cdef Py_ssize_t x_stride = x.strides[innerdim], out_stride = out.strides[innerdim]
        cdef Py_ssize_t n = x.shape[innerdim]
        cdef Py_ssize_t i
        cdef char *x_buf, *out_buf
        cdef double xval

        while np.PyArray_MultiIter_NOTDONE(mit):
            x_buf = <char*>np.PyArray_MultiIter_DATA(mit, 0)
            out_buf = <char*>np.PyArray_MultiIter_DATA(mit, 1)
            for i in range(n):
                # To access the elements we need to use some pointer arithmetic
                xval = (<double*>(x_buf + i * x_stride))[0]
                (<double*>(out_buf + i * out_stride))[0] = gsl_spline_eval(spline, xval, acc)
            np.PyArray_MultiIter_NEXT(mit)

        return out
    
    # cdef classes are not pickleable automatically, instead we must
    # provide a __reduce__ method. GSL doesn't really allow persisting
    # the spline itself so it will be recomputed when loading the pickle...
    def __reduce__(self):
        version = 0
        return (_unpickle, (version, (self.x, self.y, self.algorithm)))

def _unpickle(version, data):
    if version == 0:
        x, y, algorithm = data
        return Spline(x, y, algorithm)
    else:
        raise ValueError("Data format not supported")

@cython.boundscheck(False)
@cython.wraparound(False)
def resample(sample_x, sample_y, new_x, algorithm='cspline'):
    """
    Input: A set of samples of a function given in sample_x and sample_y.
    Output: The function resampled to the grid given in new_x.

    All inputs must be 1D arrays.
    """
    cdef Spline spline = Spline(sample_x, sample_y, algorithm=algorithm)
    cdef np.ndarray[double] new_x_arr = np.asarray(new_x, dtype=np.double)
    cdef np.ndarray[double] out = np.zeros_like(new_x_arr)
    cdef Py_ssize_t i
    for i in range(new_x_arr.shape[0]):
        out[i] = spline.evaluate_single(new_x_arr[i])
    return out


def example1():
    """
    The example from the GSL documentation. Returns a matplotlib
    figure.

    NOT ERROR-SAFE!
    """
    cdef int i
    cdef double xi, yi
    cdef np.ndarray[double] x, y
    cdef np.ndarray[double] xhigh, yhigh

    ir = np.arange(10)
    x = (ir + 0.5 * np.sin(ir)).astype(np.double)
    y = (ir + np.cos(ir*ir)).astype(np.double)

    xhigh = np.linspace(x[0], x[-1], 100)
    yhigh = np.zeros(100, dtype=np.double)
    
    cdef gsl_interp_accel *acc = gsl_interp_accel_alloc()
    cdef gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, 10)

    gsl_spline_init(spline, <double*>x.data, <double*>y.data, 10)
    for i in range(xhigh.shape[0]):
        yhigh[i] = gsl_spline_eval(spline, xhigh[i], acc)

    gsl_spline_free (spline);
    gsl_interp_accel_free (acc);

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x, y, 'r', xhigh, yhigh, 'g')
    return fig
