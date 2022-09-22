from scipy import fftpack as sp_ft
import numpy as np

def covar(im):

    class Results:
        """
        A minimal class for use when returning multiple values from a function
        This class supports dict-like assignment and retrieval
        (``obj['im'] = im``), namedtuple-like attribute look-ups (``obj.im``),
        and generic class-like object assignment (``obj.im = im``)
        """
        _value = "Description"
        _key = "Item"

        def __iter__(self):
            for item in self.__dict__.values():
                yield item

        def __getitem__(self, key):
            return getattr(self, key)

        def __setitem__(self, key, value):
            self.__dict__[key] = value

        def __str__(self):
            header = "―" * 78
            lines = [header, "{0:<25s} {1}".format(self._key, self._value), header]
            for item in list(self.__dict__.keys()):
                if item.startswith('_'):
                    continue
                if (isinstance(self[item], np.ndarray)):
                    s = np.shape(self[item])
                    if (self[item].ndim > 1):
                        lines.append("{0:<25s} Image of size {1}".format(item, s))
                    else:
                        lines.append("{0:<25s} Array of size {1}".format(item, s))
                else:
                    lines.append("{0:<25s} {1}".format(item, self[item]))
            lines.append(header)
            return "\n".join(lines)


    def _radial_profile(autocorr, r_max, nbins=100):

        if len(autocorr.shape) == 2:
            adj = np.reshape(autocorr.shape, [2, 1, 1])
            inds = np.indices(autocorr.shape) - adj / 2
            dt = np.sqrt(inds[0]**2 + inds[1]**2)
        elif len(autocorr.shape) == 3:
            adj = np.reshape(autocorr.shape, [3, 1, 1, 1])
            inds = np.indices(autocorr.shape) - adj / 2
            dt = np.sqrt(inds[0]**2 + inds[1]**2 + inds[2]**2)
        else:
            raise Exception('Image dimensions must be 2 or 3')
        bin_size = np.int(np.ceil(r_max / nbins))
        bins = np.arange(bin_size, r_max, step=bin_size)
        radial_sum = np.zeros_like(bins)
        for i, r in enumerate(bins):
            # Generate Radial Mask from dt using bins
            mask = (dt <= r) * (dt > (r - bin_size))
            radial_sum[i] = np.sum(autocorr[mask]) / np.sum(mask)
        # Return normalized bin and radially summed autoc
        norm_autoc_radial = radial_sum / np.max(autocorr)
        tpcf = Results()
        tpcf.distance = bins
        tpcf.probability = norm_autoc_radial
        return tpcf


    def two_point_correlation(im):

        # Calculate half lengths of the image
        hls = (np.ceil(np.shape(im)) / 2).astype(int)
        # Fourier Transform and shift image
        F = sp_ft.ifftshift(sp_ft.fftn(sp_ft.fftshift(im)))
        # Compute Power Spectrum
        P = np.absolute(F**2)
        # Auto-correlation is inverse of Power Spectrum
        autoc = np.absolute(sp_ft.ifftshift(sp_ft.ifftn(sp_ft.fftshift(P))))
        tpcf = _radial_profile(autoc, r_max=np.min(hls))
        return tpcf


    return two_point_correlation(im)