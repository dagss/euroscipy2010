cimport numpy as np

cdef extern from *:
    cdef double const_double "const double"

cdef extern from "gsl/gsl_interp.h":

    ctypedef struct gsl_interp_accel:
        size_t  cache
        size_t  miss_count
        size_t  hit_count

    ctypedef struct gsl_interp_type:
        char* name
        unsigned int min_size
        # we do not need to declare all fields, so skip the rest
        # for now

    gsl_interp_type * gsl_interp_linear
    gsl_interp_type * gsl_interp_polynomial
    gsl_interp_type * gsl_interp_cspline
    gsl_interp_type * gsl_interp_cspline_periodic
    gsl_interp_type * gsl_interp_akima
    gsl_interp_type * gsl_interp_akima_periodic    

    ctypedef struct gsl_interp:
        gsl_interp_type * type
        double  xmin
        double  xmax
        size_t  size
        void * state

    gsl_interp_accel * gsl_interp_accel_alloc()
    int gsl_interp_accel_reset (gsl_interp_accel * a)
    void gsl_interp_accel_free(gsl_interp_accel * a)

    gsl_interp * gsl_interp_alloc(gsl_interp_type* T, int n)
    void gsl_interp_free(gsl_interp * interp)

    double gsl_interp_eval(gsl_interp * obj, double* xa, double* ya, double x,
                           gsl_interp_accel * a)

    double gsl_interp_eval_deriv(gsl_interp * obj,
                                 double* xa, double* ya, double x,
                                 gsl_interp_accel * a)

    double gsl_interp_eval_deriv2(gsl_interp * obj,
                                  double* xa, double *ya, double x,
                                  gsl_interp_accel * a)
    
     
    int gsl_interp_init(gsl_interp * obj, double* xa, double* ya, size_t size)
    char * gsl_interp_name(gsl_interp * interp)
    unsigned int gsl_interp_min_size(gsl_interp * interp)


cdef extern from "gsl/gsl_spline.h":
    ctypedef struct gsl_spline:
        gsl_interp * interp
        double  * x
        double  * y
        size_t  size

    gsl_spline *gsl_spline_alloc(gsl_interp_type* T, size_t size)

    int gsl_spline_init(gsl_spline *spline, double *xa, double *ya, size_t size)
    char *gsl_spline_name(gsl_spline * spline)
    unsigned int gsl_spline_min_size(gsl_spline spline)

    int gsl_spline_eval_e(gsl_spline * spline, double x, gsl_interp_accel * a, double * y)

    double gsl_spline_eval(gsl_spline * spline, double x, gsl_interp_accel * a)

    int gsl_spline_eval_deriv_e(gsl_spline * spline,
                                double x,
                                gsl_interp_accel * a,
                                double * y)

    double gsl_spline_eval_deriv(gsl_spline * spline,
                                 double x,
                                 gsl_interp_accel * a)

    int gsl_spline_eval_deriv2_e(gsl_spline * spline,
                                 double x,
                                 gsl_interp_accel * a,
                                 double * y)

    double gsl_spline_eval_deriv2(gsl_spline * spline,
                                  double x,
                                  gsl_interp_accel * a)
    
    int gsl_spline_eval_integ_e(gsl_spline * spline,
                                double a, double b,
                                gsl_interp_accel * acc,
                                double * y)

    double gsl_spline_eval_integ(gsl_spline * spline,
                                 double a, double b,
                                 gsl_interp_accel * acc)
    
    void gsl_spline_free(gsl_spline * spline)
