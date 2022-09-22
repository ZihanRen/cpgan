import porespy as ps

def phi(im):
    # im: numpy array 
    phi = ps.metrics.porosity(im)
    return phi
