'''
COVMAT1D_FROM_CFUN   Correlation function based covariance matrix

   This function sets up a covariance matrix from a defined correlation 
   function. The correlation function is specified by giving a functional
   form (such as exponential decreasing) and correlation lengths. The
   correlation length is throughout defined as the distance where the 
   correlation has dropped to exp(-1). For off-diagonal values, the 
   correlation length is averaged between the two involved positions.

   Correlation matrices are obtained by setting *Std* to [].

FORMAT   S = covmat1d_from_cfun( xp, Std, cfun, Cl, [, cco, mapfun] )
       
OUT   S      The covariance matrix
IN    xp     The data abscissa.
      Std    Standard deviations. Given as a two vector matrix. First column
             holds position in same unit as *xp*. The second column is the 
             standard deviation at the postions of the first column. These
             values are then interpolated to *xp*, extrapolating end
             values to +-Inf (in a "nearest" manner).
             If set to a scalar, this value is applied for all *xp*.
             If set to [], unit standard deviation is assumed.
      cfun   Correlation function. Possible choices are
              'drc' : Dirac. No correlation. Any given correlation length
                      is ignored here.
              'lin' : Linearly decreasing (down to zero).
              'exp' : Exponential decreasing (exp(-dx/cl)).
              'gau' : Gaussian (normal) deceasing (exp(-(dx/cl))^2).
OPT   Cl     Correlation lengths. Given as a column matrix as *Std*.
             Must be given for all *cfun* beside 'drc'. Extrapolation as
             for *Std*. Scalar input is allowed.
      cco    Correlation cut-off. All values below this limit are set to 0.
      mapfun Mapping function from grid unit to unit for corrleation
             lengths. For example, if correlation lengths are given in
             pressure decades, while the basic coordinate is Pa, this is
             *mapfun* handled by setting *mapfun* to @log10. 

2005-05-20   Created by Patrick Eriksson.



2018-10-25   Below copied/modified from MATLAB to Python by B Lee
'''

import numpy as np
import numpy.matlib as npm
import scipy.interpolate as spi
import copy 

def covmat1d_from_cfun(xp1,Std1,cfun1,Cl1=None,cco1=0,mapfun1=None):
    
    xp = copy.deepcopy(xp1)
    Std = copy.deepcopy(Std1)
    cfun = copy.deepcopy(cfun1)
    Cl = copy.deepcopy(Cl1)
    cco = copy.deepcopy(cco1)
    mapfun = copy.deepcopy(mapfun1)
    
    def handle_expand(x,xi):
        
        v1 = np.min( x );
        i1 = ( xi<v1 );
        
        v2 = np.max( x );
        i2 = ( xi>v2 );
        
        if np.sum(v1) > 0:
            xi[i1,] = v1;
          
        if np.sum(v2) > 0:
            xi[i2,] = v2;
        
        return xi
    
    # Determine standard deviations
    #
    n  = len( xp );
    #
    if np.isscalar(Std):
        si = npm.repmat( Std, n, 1 );
    else:
        f = spi.interp1d(Std[:,0], Std[:,1])
        si = f(handle_expand(Std[:,0],xp));
        si = np.array([si]).T
    
    # Handle diagonal matrices separately (note return)
    #
    if cfun.lower() == 'drc':
        S = np.zeros([n,n])
        for i in range(0,n):
            S[i,i] = si[i]**2 ;
        return S
    
    # Conversion of length unit
    #
    if mapfun != None:
        if not callable(mapfun):
            Exception('Input *mapfun* must be empty or a function handle.')
        xp = mapfun(xp)
        if not np.isscalar(Cl):
            Cl[:,0] = mapfun( Cl[:,0] );
    
    # Distance matrix
    #
    [X1, X2] = npm.meshgrid(xp, xp);
    D = np.abs(X1-X2)
    
    # Correlation length matrix
    #
    if np.isscalar(Cl):
        L = Cl;
    else:
        f = spi.interp1d(Cl[:,0], Cl[:,1])
        si = f(handle_expand(handle_expand(Cl[:,0],xp)));
        [X1,X2] = npm.meshgrid( cl, cl );
        L       = ( X1 + X2 ) / 2;
    
    # Create correlation matrix
    #
    if cfun.lower() == 'lin':
        S = 1 - (1-np.exp(-1)) * ( D/L );
        # Negativa values removed by cco
    elif cfun.lower() == 'exp':
        S = np.exp(D/L)
    elif cfun.lower() == 'gau':
        S = np.exp( -(D/L)**2 );
    else:
        Exception('Unknown correlation function: ' + cfun)
    
    # Remove values below correlation cut-off limit, convert to sparse and
    # include standard deviations
    #
    S[S < cco] = 0;
    S       = (si @ si.T) * S;

    return S