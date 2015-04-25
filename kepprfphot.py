import os
import sys
import time
import glob
import pyfits
import multiprocessing
import keputils
import itertools as it
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import scipy.interpolate as interpolate
import scipy.stats as stats
import pdb

def kepprfphot(infile, outroot, prfdir, columns, rows, fluxes, border=0,
background=False, focus=False, ranges=None, xtolerance=1.e-7,
ftolerance=1.e-7, qualflags=False, plot=False, clobber=False,
verbose=False, multithreaded=False):
    '''
    Arguments:
    infile -- the name of a MAST standard format FITS file containing
        Kepler Target Pixel data within the first data extension
    outroot -- the full or relative directory path to a folder that
        will contain all output
    prfdir -- the full or relative directory path to a folder containing
        the Kepler PSF calibration
    columns -- int or list of ints specifying the starting guess for the
        CCD column position(s) of the source(s) to be fit
    rows -- int or list of ints specifying the starting guess for the
        CCD row position(s) of the source(s) to be fit
    ranges -- a list of the form [[start, stop], [start, stop], ...] or
        None to use the full time range (default)
    qualflags -- if False, archived observations flagged with any
        quality issue will not be fit
    '''
    np.seterr(all='ignore')

    if type(fluxes) is float:
        fluxes = [fluxes]

    if type(columns) is int:
        columns = [columns]

    if type(rows) is int:
        rows = [rows]

    # Determine number of sources from length of `fluxes`.
    nsrc = len(fluxes)

    if len(fluxes) != len(columns) or len(columns) != len(rows):
        raise ValueError('The lists `fluxes`, `columns`, and `rows` '
            'must all be the same length.')

    # Construct an inital guess vector for fit.
    guess = []
    guess.extend(fluxes)
    guess.extend(columns)
    guess.extend(rows)

    if background:
        if border == 0:
            guess.append(0.)
        else:
            guess.extend([0.] * (border + 1) * 2)

    if focus:
        guess.extend([1., 1., 0.])

    # Clobber all output files; raises an error if any output file
    # already exists.
    for i in xrange(nsrc):
        outfile = '%s_%d.fits' % (outroot, i)

        if clobber:
            try:
                os.remove(outfile)
            except OSError:
                # The file does not already exist; clobbering will have
                # no effect.
                pass

        if os.path.isfile(outfile):
            raise RuntimeError('Output file ' + outfile + ' exists; use '
                'clobber=True to overwrite.')

    # Open the TPF FITS file and get header/column data.
    colnames = ['TIME', 'TIMECORR', 'CADENCENO', 'FLUX', 'FLUX_ERR',
        'POS_CORR1', 'POS_CORR2', 'QUALITY']
    kepid, channel, skygroup, module, output, quarter, season, \
        ra, dec, column, row, kepmag, xdim, ydim, pixels = \
        keputils.read_tpf(infile, colnames)

    barytime = pixels[0]
    tcorr = pixels[1]
    cadno = pixels[2]
    fluxpixels = pixels[3]
    errpixels = pixels[4]
    poscorr1 = pixels[5]
    poscorr2 = pixels[6]
    qual = pixels[7]

    # Open the input file and get the time keywords.
    struct = pyfits.open(infile, mode='readonly')
    tstart, tstop, bjdref, cadence = keputils.timekeys(struct)

    # Generate the mask map.
    cards0 = struct[0].header.ascardlist()
    cards1 = struct[1].header.ascardlist()
    cards2 = struct[2].header.ascardlist()
    maskmap = struct[2].data.copy()
    npix = np.size(np.nonzero(maskmap)[0])

    # Determine suitable PRF calibration file.
    prfglob = prfdir + '/' + ('kplr%02i' % module) + '.' + \
        str(output) + '*' + '_prf.fits'
    prffile = glob.glob(prfglob)[0]

    # Read the PRF images.
    prfn = [None, None, None, None, None]
    crpix1p = np.empty((5,), dtype='float64')
    crpix2p = np.empty((5,), dtype='float64')
    crval1p = np.empty((5,), dtype='float64')
    crval2p = np.empty((5,), dtype='float64')
    cdelt1p = np.empty((5,), dtype='float64')
    cdelt2p = np.empty((5,), dtype='float64')

    for i in xrange(5):
        prfn[i], crpix1p[i], crpix2p[i], crval1p[i], crval2p[i], \
            cdelt1p[i], cdelt2p[i] = keputils.read_prf_image(prffile, i + 1)

    PRFx = np.arange(0.5, np.shape(prfn[0])[1] + 0.5)
    PRFy = np.arange(0.5, np.shape(prfn[0])[0] + 0.5)
    PRFx = (PRFx - np.size(PRFx) / 2) * cdelt1p[0]
    PRFy = (PRFy - np.size(PRFy) / 2) * cdelt2p[0]

    # Interpolate the calibrated PRF shape to the target position.
    prf = np.zeros(np.shape(prfn[0]), dtype='float64')
    prfWeight = np.zeros((5,), dtype='float64')

    for i in xrange(5):
        prfWeight[i] = np.sqrt((column - crval1p[i])**2. +
            (row - crval2p[i])**2.)

        if prfWeight[i] == 0.:
            prfWeight[i] = 1.e6

        prf = prf + prfn[i] / prfWeight[i]

    prf = prf / np.nansum(prf)
    prf = prf / cdelt1p[0] / cdelt2p[0]

    # Location of the data image centered on the PRF image (in PRF
    # pixel units)
    prfDimY = ydim / cdelt1p[0]
    prfDimX = xdim / cdelt2p[0]
    PRFy0 = (np.shape(prf)[0] - prfDimY) / 2
    PRFx0 = (np.shape(prf)[1] - prfDimX) / 2

    # Construct the input pixel image.
    DATx = np.arange(column, column + xdim)
    DATy = np.arange(row, row + ydim)

    # Interpolation function over the PRF
    splineInterpolation = interpolate.RectBivariateSpline(PRFx, PRFy,
        prf, kx=3, ky=3)

    # Construct a mesh for the background model.
    bx = np.arange(1., float(xdim + 1))
    by = np.arange(1., float(ydim + 1))
    xx, yy = np.meshgrid(np.linspace(np.amin(bx), np.amax(bx), xdim),
        np.linspace(np.amin(by), np.amax(by), ydim))

    # Get time ranges for new photometry and flag good data.
    barytime += bjdref
    incl = np.empty((np.size(barytime),), dtype='int32')

    if ranges is None:
        ranges = [[np.nanmin(barytime), np.nanmax(barytime)]]

    for rownum in xrange(np.size(barytime)):
        for winnum in xrange(len(ranges)):
            if barytime[rownum] >= ranges[winnum][0] and \
                barytime[rownum] <= ranges[winnum][1] and \
                (qual[rownum] == 0 or qualflags) and \
                np.isfinite(barytime[rownum]) and \
                np.isfinite(np.nansum(fluxpixels[rownum,:])):
                    incl[rownum] = 1

    if not np.in1d(1, incl):
        raise RuntimeError('No legal data within the range ' + str(ranges) + '.')

    # Filter out bad data.
    ndx = np.where(incl == 1)
    nincl = np.size(ndx)
    barytime = barytime[ndx]
    tcorr = tcorr[ndx]
    fluxpixels = fluxpixels[ndx]
    errpixels = errpixels[ndx]
    poscorr1 = poscorr1[ndx]
    poscorr2 = poscorr2[ndx]
    qual = qual[ndx]

    if verbose:
        sys.stdout.write('Preparing...')
        sys.stdout.flush()

    oldtime = 0.

    for rownum in xrange(min(80, np.size(barytime))):
        if barytime[rownum] - oldtime > 0.5:
            ftol = 1.e-10
            xtol = 1.e-10

        guess = __prf_fits(fluxpixels[rownum,:], errpixels[rownum,:],
            DATx, DATy, nsrc, border, xx, yy, PRFx, PRFy,
            splineInterpolation, guess, ftol, xtol, focus, background,
            rownum, 80, float(columns[0]), float(rows[0]), False)

        ftol = ftolerance
        xtol = xtolerance
        oldtime = barytime[rownum]

    # Do the PSF fitting.
    if multithreaded:
        ans = []
        cad1 = 0
        cad2 = 50

        for i in xrange(int(nincl / 50) + 1):
            try:
                fluxp = fluxpixels[cad1:cad2,:]
                errp = errpixels[cad1:cad2,:]
                progress = np.arange(cad1,cad2)
            except IndexError:
                fluxp = fluxpixels[cad1:nincl,:]
                errp = errpixels[cad1:nincl,:]
                progress = np.arange(cad1,nincl)

            args = itertools.izip(fluxp, errp, it.repeat(DATx),
                it.repeat(DATy), it.repeat(nsrc), it.repeat(border),
                it.repeat(xx), it.repeat(yy), it.repeat(PRFx),
                it.repeat(PRFy), it.repeat(splineInterpolation),
                it.repeat(guess), it.repeat(ftolerance), it.repeat(xtolerance),
                it.repeat(focus), it.repeat(background), progress,
                it.repeat(np.arange(cad1, nincl)[-1]), it.repeat(columns[0]),
                it.repeat(rows[0]), it.repeat(verbose))

            p = multiprocessing.Pool()
            model = p.imap(__prf_fits, args, chunksize=1)
            p.close()
            p.join()

            cad1 += 50
            cad2 += 50

            # Use `extend` here because `model` will be a list of lists.
            ans.extend(model)
            guess = anslist[-1]

        ans = np.array(ans, dtype='float64').transpose()
    else:
        oldtime = 0.
        ans = []

        for rownum in xrange(nincl):
            if barytime[rownum] - oldtime > 0.5:
                ftol = 1.e-10
                xtol = 1.e-10

            guess = __prf_fits(fluxpixels[rownum,:], errpixels[rownum,:],
                DATx, DATy, nsrc, border, xx, yy, PRFx, PRFy,
                splineInterpolation, guess, ftol, xtol, focus, background,
                rownum, nincl, columns[0], rows[0], verbose)
            #print guess

            # Use `append` here because `guess` is a list of parameters.
            ans.append(guess)

            ftol = ftolerance
            xtol = xtolerance
            oldtime = barytime[rownum]

        ans = np.array(ans, dtype='float64').transpose()

    # Unpack the best fit parameters.
    flux = []
    OBJx = []
    OBJy = []
    na = np.shape(ans)[1]

    for i in xrange(nsrc):
        flux.append(ans[i,:])
        OBJx.append(ans[nsrc+i,:])
        OBJy.append(ans[nsrc*2+i,:])

    try:
        bterms = border + 1

        if bterms == 1:
            b = ans[nsrc*3,:]
        else:
            b = np.array([], dtype='float64')
            bkg = []

            for i in xrange(na):
                bcoeff = np.array([ans[nsrc*3:nsrc*3+bterms,i],
                    ans[nsrc*3+bterms:nsrc*3+bterms*2,i]], dtype='float64')
                bkg.append(__polyval2d(xx, yy, bcoeff))
                b = np.append(b, stats.nanmean(bkg[-1].reshape(bkg[-1].size)))
    except IndexError:
        b = np.zeros((na,), dtype='float64')

    if focus:
        wx = ans[-3,:]
        wy = ans[-2,:]
        angle = ans[-1,:]
    else:
        wx = np.ones((na,), dtype='float64')
        wy = np.ones((na,), dtype='float64')
        angle = np.zeros((na,), dtype='float64')

    # Constuct model PRF in detector coordinates.
    residual = []
    chi2 = []

    for i in xrange(na):
        f = np.empty((nsrc,), dtype='float64')
        x = np.empty((nsrc,), dtype='float64')
        y = np.empty((nsrc,), dtype='float64')

        for j in xrange(nsrc):
            f[j] = flux[j][i]
            x[j] = OBJx[j][i]
            y[j] = OBJy[j][i]

        PRFfit = __prf_to_detector(f, x, y, DATx, DATy, wx[i], wy[i],
            angle[i], splineInterpolation)

        if background and bterms == 1:
            PRFfit = PRFfit + b[i]

        if background and bterms > 1:
            PRFfit = PRFfit + bkg[i]

        # Calculate the residual of DATA - FIT.
        xdim = np.shape(xx)[1]
        ydim = np.shape(yy)[0]
        DATimg = fluxpixels[i].copy().reshape((ydim, xdim))
        PRFres = DATimg - PRFfit
        residual.append(np.nansum(PRFres) / npix)

        # Calculate the sum squared difference between the data and model.
        chi2.append(np.absolute(np.nansum(((DATimg - PRFfit) / PRFfit)**2.)))

    # Construct the output arrays.
    otime = barytime - bjdref
    otimecorr = tcorr
    ocadenceno = cadno
    opos_corr1 = poscorr1
    opos_corr2 = poscorr2
    oquality = qual
    opsf_bkg = b
    opsf_focus1 = wx
    opsf_focus2 = wy
    opsf_rotation = angle
    opsf_residual = residual
    opsf_chi2 = chi2

    opsf_flux_err = np.empty((na,), dtype='float64')
    opsf_flux_err.fill(np.nan)

    opsf_centr1_err = np.empty((na,), dtype='float64')
    opsf_centr1_err.fill(np.nan)

    opsf_centr2_err = np.empty((na,), dtype='float64')
    opsf_centr2_err.fill(np.nan)

    opsf_bkg_err = np.empty((na,), dtype='float64')
    opsf_bkg_err.fill(np.nan)

    opsf_flux = flux
    opsf_centr1 = OBJx
    opsf_centr2 = OBJy

    # Construct the output primary extension.
    for j in range(nsrc):
        hdu0 = pyfits.PrimaryHDU()

        for i in xrange(len(cards0)):
            if cards0[i].key not in hdu0.header.ascardlist().keys():
                hdu0.header.update(cards0[i].key, cards0[i].value, cards0[i].comment)
            else:
                hdu0.header.ascardlist()[cards0[i].key].comment = cards0[i].comment

        outstr = pyfits.HDUList(hdu0)

        # Construct the output light curve extension.
        col1 = pyfits.Column(name='TIME', format='D', unit='BJD - 2454833', array=otime)
        col2 = pyfits.Column(name='TIMECORR', format='E', unit='d', array=otimecorr)
        col3 = pyfits.Column(name='CADENCENO', format='J', array=ocadenceno)
        col4 = pyfits.Column(name='PSF_FLUX', format='E', unit='e-/s', array=opsf_flux[j])
        col5 = pyfits.Column(name='PSF_FLUX_ERR', format='E', unit='e-/s', array=opsf_flux_err)
        col6 = pyfits.Column(name='PSF_BKG', format='E', unit='e-/s', array=opsf_bkg)
        col7 = pyfits.Column(name='PSF_BKG_ERR', format='E', unit='e-/s', array=opsf_bkg_err)
        col8 = pyfits.Column(name='PSF_CENTR1', format='E', unit='pixel', array=opsf_centr1[j])
        col9 = pyfits.Column(name='PSF_CENTR1_ERR', format='E', unit='pixel', array=opsf_centr1_err)
        col10 = pyfits.Column(name='PSF_CENTR2', format='E', unit='pixel', array=opsf_centr2[j])
        col11 = pyfits.Column(name='PSF_CENTR2_ERR', format='E', unit='pixel', array=opsf_centr2_err)
        col12 = pyfits.Column(name='PSF_FOCUS1', format='E', unit='e-/s', array=opsf_focus1)
        col13 = pyfits.Column(name='PSF_FOCUS2', format='E', unit='e-/s', array=opsf_focus2)
        col14 = pyfits.Column(name='PSF_ROTATION', format='E', unit='deg', array=opsf_rotation)
        col15 = pyfits.Column(name='PSF_RESIDUAL', format='E', unit='e-/s', array=opsf_residual)
        col16 = pyfits.Column(name='PSF_CHI2', format='E', array=opsf_chi2)
        col17 = pyfits.Column(name='POS_CORR1', format='E', unit='pixel', array=opos_corr1)
        col18 = pyfits.Column(name='POS_CORR2', format='E', unit='pixel', array=opos_corr2)
        col19 = pyfits.Column(name='SAP_QUALITY', format='J', array=oquality)
        cols = pyfits.ColDefs([col1, col2, col3, col4, col5, col6, col7,
            col8, col9, col10, col11, col12, col13, col14, col15, col16,
            col17, col18, col19])
        hdu1 = pyfits.new_table(cols)

        for i in xrange(len(cards1)):
            if (cards1[i].key not in hdu1.header.ascardlist().keys() and
            cards1[i].key[:4] not in ['TTYP', 'TFOR', 'TUNI', 'TDIS',
            'TDIM', 'WCAX', '1CTY', '2CTY', '1CRP', '2CRP', '1CRV',
            '2CRV', '1CUN', '2CUN', '1CDE', '2CDE', '1CTY', '2CTY',
            '1CDL', '2CDL', '11PC', '12PC', '21PC', '22PC']):
                hdu1.header.update(cards1[i].key, cards1[i].value,
                    cards1[i].comment)

        outstr.append(hdu1)

        # Construct the output mask bitmap extension.
        hdu2 = pyfits.ImageHDU(maskmap)

        for i in xrange(len(cards2)):
            if cards2[i].key not in hdu2.header.ascardlist().keys():
                hdu2.header.update(cards2[i].key, cards2[i].value,
                    cards2[i].comment)
            else:
                hdu2.header.ascardlist()[cards2[i].key].comment = \
                    cards2[i].comment

        outstr.append(hdu2)

        # Write the output file and close.
        outstr.writeto(outroot + '_' + str(j) + '.fits', checksum=True)
        outstr.close()

    __do_plots(barytime, flux, OBJx, OBJy, b, wx, wy, angle, residual,
        chi2, nsrc, cadence, focus, background, npix, guess, outroot,
        show=plot)


def __do_plots(t, fl, dx, dy, bg, fx, fy, fa, rs, ch, nsrc, cadence,
focus, background, npix, guess, outroot, show=False):
    '''
    t -- time
    fl -- flux
    rs -- residual
    ch -- chi^2 (sum of the squared residuals)
    '''
    # Clean up x-axis units.
    barytime0 = float(int(t[0] / 100) * 100.)
    t -= barytime0
    xlab = 'BJD $-$ %d' % barytime0

    # Set the plot style.
    #mpl.rcParams['axes.linewidth'] = 2.5
    #mpl.rcParams['axes.labelsize'] = 24
    #mpl.rcParams['font.family'] = 'sans-serif'
    #mpl.rcParams['font.weight'] = 'bold'
    #mpl.rcParams['font.size'] = 12
    #mpl.rcParams['legend.fontsize'] = 12
    #mpl.rcParams['xtick.labelsize'] = 12
    #mpl.rcParams['ytick.labelsize'] = 12

    for i in range(nsrc):
        # Clean up y-axis units.
        nrm = np.ceil(np.log10(np.nanmax(fl[i]))) - 1.
        fl[i] /= 10.**nrm

        if nrm == 0:
            ylab1 = 'e$^-$ s$^{-1}$'
        else:
            ylab1 = '10$^{%d}$ e$^-$ s$^{-1}$' % nrm

        xx = np.copy(dx[i])
        yy = np.copy(dy[i])
        ylab2 = 'offset (pixels)'

        # Set data limits.
        xmin = np.nanmin(t)
        xmax = np.nanmax(t)
        ymin1 = np.nanmin(fl[i])
        ymax1 = np.nanmax(fl[i])
        ymin2 = np.nanmin(xx)
        ymax2 = np.nanmax(xx)
        ymin3 = np.nanmin(yy)
        ymax3 = np.nanmax(yy)
        ymin4 = np.nanmin(bg[1:-1])
        ymax4 = np.nanmax(bg[1:-1])
        ymin5 = np.nanmin([np.nanmin(fx), np.nanmin(fy)])
        ymax5 = np.nanmax([np.nanmax(fx), np.nanmax(fy)])
        ymin6 = np.nanmin(fa[1:-1])
        ymax6 = np.nanmax(fa[1:-1])
        ymin7 = np.nanmin(rs[1:-1])
        ymax7 = np.nanmax(rs[1:-1])
        ymin8 = np.nanmin(ch[1:-1])
        ymax8 = np.nanmax(ch[1:-1])
        xr = xmax - xmin
        yr1 = ymax1 - ymin1
        yr2 = ymax2 - ymin2
        yr3 = ymax3 - ymin3
        yr4 = ymax4 - ymin4
        yr5 = ymax5 - ymin5
        yr6 = ymax6 - ymin6
        yr7 = ymax7 - ymin7
        yr8 = ymax8 - ymin8

        # Position first axes inside the plotting window.
        ax = plt.axes([0.11, 0.523, 0.78, 0.45])

        # Force tick labels to be absolute rather than relative.
        plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))
        plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))

        # We don't want an x-label.
        plt.setp(plt.gca(), xticklabels=[])

        # Plot flux vs. time.
        ltime = np.array([],dtype='float64')
        ldata = np.array([],dtype='float32')
        dt = 0
        work1 = 2. * cadence / 86400.

        for j in xrange(1, len(t) - 1):
            dt = t[j] - t[j-1]

            if dt < work1:
                ltime = np.append(ltime,t[j])
                ldata = np.append(ldata,fl[i][j])
            else:
                plt.plot(ltime, ldata, color='blue', linestyle='-', linewidth=1.)
                ltime = np.array([],dtype='float64')
                ldata = np.array([],dtype='float64')

        plt.plot(ltime, ldata, color='blue', linestyle='-', linewidth=1.)
        plt.fill_between(t, fl[i], facecolor='yellow', linewidth=0.,
            alpha=0.2, interpolate=True)

        # Define plot x and y limits.
        plt.xlim(xmin - xr * 0.01, xmax + xr * 0.01)

        if ymin1 - yr1 * 0.01 <= 0.0:
            plt.ylim(1.e-10, ymax1 + yr1 * 0.01)
        else:
            plt.ylim(ymin1 - yr1 * 0.01, ymax1 + yr1 * 0.01)

        # Set plot labels and turn on grid.
        plt.ylabel('Source (' + ylab1 + ')', color='black')
        plt.grid()

        # Plot centroid tracks and position second axes inside the
        # plotting window.
        if focus and background:
            axs = [0.11, 0.433, 0.78, 0.09]
        elif background or focus:
            axs = [0.11, 0.388, 0.78, 0.135]
        else:
            axs = [0.11, 0.253, 0.78, 0.27]

        ax1 = plt.axes(axs)

        # Force tick labels to be absolute rather than relative.
        plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))
        plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))
        plt.setp(plt.gca(), xticklabels=[])

        # Plot dx vs. time.
        ltime = np.array([],dtype='float64')
        ldata = np.array([],dtype='float64')
        dt = 0.
        work1 = 2. * cadence / 86400.

        for j in xrange(1, len(t) - 1):
            dt = t[j] - t[j-1]

            if dt < work1:
                ltime = np.append(ltime, t[j])
                ldata = np.append(ldata, xx[j-1])
            else:
                ax1.plot(ltime, ldata, color='red', linestyle='-', linewidth=1.)
                ltime = np.array([],dtype='float64')
                ldata = np.array([],dtype='float64')

        ax1.plot(ltime, ldata, color='red', linestyle='-', linewidth=1.,
            label='dx')

        # Define plot x and y limits.
        plt.xlim(xmin - xr * 0.01, xmax + xr * 0.01)
        plt.ylim(ymin2 - yr2 * 0.03, ymax2 + yr2 * 0.03)

        # Set plot labels.
        ax1.set_ylabel('X-' + ylab2, color='k', fontsize=11)

        # Position second axes inside the plotting window.
        ax2 = ax1.twinx()

        # Force tick labels to be absolute rather than relative.
        plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))
        plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))
        plt.setp(plt.gca(), xticklabels=[])

        # Plot dy vs. time.
        ltime = np.array([],dtype='float64')
        ldata = np.array([],dtype='float32')
        dt = 0.
        work1 = 2. * cadence / 86400.

        for j in xrange(1, len(t) - 1):
            dt = t[j] - t[j-1]

            if dt < work1:
                ltime = np.append(ltime, t[j])
                ldata = np.append(ldata, yy[j-1])
            else:
                ax2.plot(ltime, ldata, color='green', linestyle='-', linewidth=1.)
                ltime = np.array([],dtype='float64')
                ldata = np.array([],dtype='float64')

        ax2.plot(ltime, ldata, color='green', linestyle='-', linewidth=1.,
            label='dy')

        # Define plot y limits.
        plt.xlim(xmin - xr * 0.01, xmax + xr * 0.01)
        plt.ylim(ymin3 - yr3 * 0.03, ymax3 + yr3 * 0.03)

        # Set plot labels.
        ax2.set_ylabel('Y-' + ylab2, color='k', fontsize=11)

        # Position third axes inside the plotting window.
        if background and focus:
            axs = [0.11, 0.343, 0.78, 0.09]

        if background and not focus:
            axs = [0.11, 0.253, 0.78, 0.135]

        if background:
            ax1 = plt.axes(axs)

            # Force tick labels to be absolute rather than relative.
            plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))
            plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))
            plt.setp(plt.gca(), xticklabels=[])

            # Plot background vs. time.
            ltime = np.array([],dtype='float64')
            ldata = np.array([],dtype='float32')
            dt = 0.
            work1 = 2. * cadence / 86400.

            for j in xrange(1, len(t) - 1):
                dt = t[j] - t[j-1]

                if dt < work1:
                    ltime = np.append(ltime, t[j])
                    ldata = np.append(ldata, bg[j])
                else:
                    ax1.plot(ltime, ldata, color='blue', linestyle='-', linewidth=1.)
                    ltime = np.array([],dtype='float64')
                    ldata = np.array([],dtype='float64')

            # Plot the fill color below data time series, with no data gaps.
            ax1.plot(ltime, ldata, color='blue', linestyle='-', linewidth=1.)
            plt.fill_between(t, bg, facecolor='yellow', linewidth=0.,
                alpha=0.2, interpolate=True)

            # Define plot x and y limits.
            plt.xlim(xmin - xr * 0.01, xmax + xr * 0.01)
            plt.ylim(ymin4 - yr4 * 0.03, ymax4 + yr4 * 0.03)

            # Set plot labels and turn on grid.
            ax1.set_ylabel('Background \n(e$^-$ s$^{-1}$ pix$^{-1}$)',
                multialignment='center', color='k', fontsize=11)
            plt.grid()

        # Position focus axes inside the plotting window.
        if focus and background:
            axs = [0.11, 0.253, 0.78, 0.09]

        if focus and not background:
            axs = [0.11, 0.253, 0.78, 0.135]

        if focus:
            ax1 = plt.axes(axs)

            # Force tick labels to be absolute rather than relative.
            plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))
            plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))
            plt.setp(plt.gca(), xticklabels=[])

            # Plot x-axis PSF width vs time.
            ltime = np.array([],dtype='float64')
            ldata = np.array([],dtype='float64')
            dt = 0.
            work1 = 2. * cadence / 86400.

            for j in xrange(1, len(t) - 1):
                dt = t[j] - t[j-1]

                if dt < work1:
                    ltime = np.append(ltime, t[j])
                    ldata = np.append(ldata, fx[j])
                else:
                    ax1.plot(ltime, ldata, color='r', linestyle='-', linewidth=1.)
                    ltime = np.array([], dtype='float64')
                    ldata = np.array([], dtype='float64')

            ax1.plot(ltime, ldata, color='r', linestyle='-', linewidth=1.)

            # Plot y-axis PSF width vs time.
            ltime = np.array([], dtype='float64')
            ldata = np.array([], dtype='float64')
            dt = 0.
            work1 = 2. * cadence / 86400.

            for j in xrange(1, len(t) - 1):
                dt = t[j] - t[j-1]

                if dt < work1:
                    ltime = np.append(ltime, t[j])
                    ldata = np.append(ldata, fy[j])
                else:
                    ax1.plot(ltime, ldata, color='g', linestyle='-', linewidth=1.)
                    ltime = np.array([], dtype='float64')
                    ldata = np.array([], dtype='float64')

            ax1.plot(ltime, ldata, color='g', linestyle='-', linewidth=1.)

            # Define plot x and y limits.
            plt.xlim(xmin - xr * 0.01, xmax + xr * 0.01)
            plt.ylim(ymin5 - yr5 * 0.03, ymax5 + yr5 * 0.03)

            # Set plot labels.
            ax1.set_ylabel('Pixel Scale\nFactor', multialignment='center',
                color='k', fontsize=11)

            # Position second axes inside the plotting window.
            ax2 = ax1.twinx()

            # Force tick labels to be absolute rather than relative.
            plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))
            plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))
            plt.setp(plt.gca(), xticklabels=[])

            # Plot dy vs. time.
            ltime = np.array([], dtype='float64')
            ldata = np.array([], dtype='float64')
            dt = 0.
            work1 = 2. * cadence / 86400.

            for j in xrange(1, len(t) - 1):
                dt = t[j] - t[j-1]

                if dt < work1:
                    ltime = np.append(ltime, t[j])
                    ldata = np.append(ldata, fa[j])
                else:
                    ax2.plot(ltime, ldata, color='#000080', linestyle='-', linewidth=1.)
                    ltime = np.array([], dtype='float64')
                    ldata = np.array([], dtype='float64')

            ax2.plot(ltime, ldata, color='#000080', linestyle='-', linewidth=1.)

            # Define plot y limits.
            plt.xlim(xmin - xr * 0.01, xmax + xr * 0.01)
            plt.ylim(ymin6 - yr6 * 0.03, ymax6 + yr6 * 0.03)

            # Set plot labels.
            ax2.set_ylabel('Rotation (deg)', color='k', fontsize=11)

        # Position fifth axes inside the plotting window.
        axs = [0.11, 0.163, 0.78, 0.09]
        ax1 = plt.axes(axs)

        # Force tick labels to be absolute rather than relative.
        plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))
        plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))
        plt.setp(plt.gca(),xticklabels=[])

        # Plot residual vs time.
        ltime = np.array([],dtype='float64')
        ldata = np.array([],dtype='float32')
        dt = 0.
        work1 = 2. * cadence / 86400.

        for j in xrange(1, len(t) - 1):
            dt = t[j] - t[j-1]

            if dt < work1:
                ltime = np.append(ltime, t[j])
                ldata = np.append(ldata, rs[j])
            else:
                ax1.plot(ltime, ldata, color='b', linestyle='-', linewidth=1.)
                ltime = np.array([], dtype='float64')
                ldata = np.array([], dtype='float64')

        # Plot the fill color below data time series, with no data gaps.
        ax1.plot(ltime, ldata, color='b', linestyle='-', linewidth=1.)
        plt.fill_between(t, rs, facecolor='yellow', linewidth=0.,
            alpha=0.2, interpolate=True)

        # Define plot x and y limits
        plt.xlim(xmin - xr * 0.01, xmax + xr * 0.01)
        plt.ylim(ymin7 - yr7 * 0.03, ymax7 + yr7 * 0.03)

        # Set plot labels and turn on grid.
        ax1.set_ylabel('Residual \n(e$^-$ s$^{-1}$)',
            multialignment='center', color='black', fontsize=11)
        plt.grid()

        # Position sixth axes inside the plotting window.
        axs = [0.11, 0.073, 0.78, 0.09]
        ax1 = plt.axes(axs)

        # Force tick labels to be absolute rather than relative.
        plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))
        plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))

        # Plot background vs. time.
        ltime = np.array([], dtype='float64')
        ldata = np.array([], dtype='float64')
        dt = 0.
        work1 = 2. * cadence / 86400.

        for j in xrange(1, len(t) - 1):
            dt = t[j] - t[j-1]

            if dt < work1:
                ltime = np.append(ltime, t[j])
                ldata = np.append(ldata, ch[j])
            else:
                ax1.plot(ltime, ldata, color='b', linestyle='-', linewidth=1.)
                ltime = np.array([], dtype='float64')
                ldata = np.array([], dtype='float64')

        # Plot the fill color below data time series, with no data gaps.
        ax1.plot(ltime, ldata, color='b', linestyle='-', linewidth=1.)
        plt.fill_between(t, ch, facecolor='yellow', linewidth=0.,
            alpha=0.2, interpolate=True)

        # Define plot x and y limits.
        plt.xlim(xmin - xr * 0.01, xmax + xr * 0.01)
        plt.ylim(ymin8 - yr8 * 0.03, ymax8 + yr8 * 0.03)

        # Set plot labels and turn on grid.
        ax1.set_ylabel('$\chi^2$ (%d dof)' % (npix - len(guess) - 1),
            color='k', fontsize=11)
        plt.xlabel(xlab, color='black')
        plt.grid()

        # Render plot.
        plt.savefig(outroot + '_' + str(i) + '.png')

#        if plt:
#            plt.show()

    mpl.rcdefaults()


def __prf_fits(fluxpixels, errpixels, DATx, DATy, nsrc, border, xx, yy,
PRFx, PRFy, splineInterpolation, guess, ftol, xtol, focus, background,
rownum, nrows, x, y, verbose):
    '''
    Fit the data to a model.
    '''
    # Save the start time.
    proctime = time.time()

    # Extract the image from the time series.
    xdim = np.shape(xx)[1]
    ydim = np.shape(xx)[0]
    DATimg = fluxpixels.copy().reshape((ydim, xdim))
    DATerr = errpixels.copy().reshape((ydim, xdim))

    # Minimize data and model.
    if focus and background:
        argm = (DATx, DATy, DATimg, DATerr, nsrc, border, xx, yy,
            splineInterpolation, x, y)
        ans = optimize.fmin(__prf_with_focus_and_background,
            guess, args=argm, xtol=xtol, ftol=ftol,disp=False)
    elif focus and not background:
        argm = (DATx, DATy, DATimg, DATerr, nsrc, splineInterpolation, x, y)
        ans = optimize.fmin(__prf_with_focus, guess, args=argm,
            xtol=xtol, ftol=ftol,disp=False)
    elif background and not focus:
        argm = (DATx, DATy, DATimg, DATerr, nsrc, border, xx, yy,
            splineInterpolation, x, y)
        ans = optimize.fmin(__prf_with_background, guess,
            args=argm, xtol=xtol, ftol=ftol, disp=False)
    else:
        argm = (DATx, DATy, DATimg, DATerr, nsrc, splineInterpolation, x, y)
        ans = optimize.fmin(__prf, guess, args=argm,
            xtol=xtol, ftol=ftol, disp=False)

    # Print progress.
    if verbose:
        txt  = '\r%3d%% ' % ((float(rownum) + 1.) / float(nrows) * 100.)
        txt += 'nrow = %d ' % (rownum + 1)
        txt += 't = %.1f sec' % (time.time() - proctime)
        txt += ' ' * 5
        sys.stdout.write(txt)
        sys.stdout.flush()

    return ans


def __prf(params, *args):
    '''
    PRF model.
    '''
    # Unpack arguments under friendly names.
    DATx = args[0]
    DATy = args[1]
    DATimg = args[2]
    DATerr = args[3]
    nsrc = args[4]
    splineInterpolation = args[5]
    col = args[6]
    row = args[7]

    # Define parameters.
    f = np.empty((nsrc,), dtype='float64')
    x = np.empty((nsrc,), dtype='float64')
    y = np.empty((nsrc,), dtype='float64')

    for i in xrange(nsrc):
        f[i] = params[i]
        x[i] = params[nsrc+i]
        y[i] = params[nsrc*2+i]

    # Calculate PRF model binned to the detector pixel size.
    PRFfit = __prf_to_detector(f, x, y, DATx, DATy, 1., 1., 0.,
        splineInterpolation)

    # Calculate the sum squared difference between data and model.
    PRFres = np.nansum((DATimg - PRFfit)**2. / DATerr**2.)

    # Keep the fit centered.
    if max(abs(col - x[0]),abs(row - y[0])) > 5.:
        PRFres = 1.e300

    return PRFres


def __prf_with_background(params, *args):
    '''
    Calculate a PRF model with a variable background.
    '''
    # Unpack arguments under friendly names.
    DATx = args[0]
    DATy = args[1]
    DATimg = args[2]
    DATerr = args[3]
    nsrc = args[4]
    bterms = args[5] + 1
    bx = args[6]
    by = args[7]
    splineInterpolation = args[8]
    col = args[9]
    row = args[10]

    # Generate parameters.
    f = np.empty((nsrc,), dtype='float64')
    x = np.empty((nsrc,), dtype='float64')
    y = np.empty((nsrc,), dtype='float64')

    for i in xrange(nsrc):
        f[i] = params[i]
        x[i] = params[nsrc+i]
        y[i] = params[nsrc*2+i]

    b = np.array([params[nsrc*3:nsrc*3+bterms],
        params[nsrc*3+bterms:nsrc*3+bterms*2]], dtype='float64')

    # Calculate PRF model binned to the detector pixel size.
    PRFfit = __prf_to_detector(f, x, y, DATx, DATy, 1., 1., 0.,
        splineInterpolation)

    # Add background.
    if bterms == 1:
        PRFfit += params[nsrc*3]
    else:
        PRFfit += __polyval2d(bx, by, b)

    # Calculate the sum squared difference between data and model.
    PRFres = np.nansum((DATimg - PRFfit)**2. / DATerr**2.)

    # Keep the fit centered.
    if max(abs(col - x[0]), abs(row - y[0])) > 5.:
        PRFres = 1.e300

    return PRFres


def __prf_with_focus(params, *args):
    '''
    PRF model with variable focus.
    '''
    # Unpack arguments under friendly names.
    DATx = args[0]
    DATy = args[1]
    DATimg = args[2]
    DATerr = args[3]
    nsrc = args[4]
    splineInterpolation = args[5]
    col = args[6]
    row = args[7]

    # Define parameters.
    f = np.empty((nsrc,), dtype='float64')
    x = np.empty((nsrc,), dtype='float64')
    y = np.empty((nsrc,), dtype='float64')

    for i in xrange(nsrc):
        f[i] = params[i]
        x[i] = params[nsrc+i]
        y[i] = params[nsrc*2+i]

    wx = params[-3]
    wy = params[-2]
    a = params[-1]

    # Calculate PRF model binned to the detector pixel size.
    PRFfit = __prf_to_detector(f, x, y, DATx, DATy, wx, wy, 0.,
        splineInterpolation)

    # Calculate the sum squared difference between data and model.
    PRFres = np.nansum((DATimg - PRFfit)**2. / DATerr**2.)

    # Keep the fit centered.
    if max(abs(col - x[0]), abs(row - y[0])) > 5.:
        PRFres = 1.e300

    return PRFres


def __prf_with_focus_and_background(params, *args):
    '''
    PRF model with variable focus and background
    '''
    # Unpack arguments under friendly names.
    DATx = args[0]
    DATy = args[1]
    DATimg = args[2]
    DATerr = args[3]
    nsrc = args[4]
    bterms = args[5] + 1
    bx = args[6]
    by = args[7]
    splineInterpolation = args[8]
    col = args[9]
    row = args[10]

    # Define parameters.
    f = np.empty((nsrc,), dtype='float64')
    x = np.empty((nsrc,), dtype='float64')
    y = np.empty((nsrc,), dtype='float64')

    for i in xrange(nsrc):
        f[i] = params[i]
        x[i] = params[nsrc+i]
        y[i] = params[nsrc*2+i]

    if bterms == 1:
        b = params[nsrc*3]
    else:
        b = np.array([params[nsrc*3:nsrc*3+bterms],
            params[nsrc*3+bterms:nsrc*3+bterms*2]], dtype='float64')

    wx = params[-3]
    wy = params[-2]
    a = params[-1]

    # Calculate PRF model binned to the detector pixel size.
    PRFfit = __prf_to_detector(f, x, y, DATx, DATy, wx, wy, a,
        splineInterpolation)

    # Add background.
    if bterms == 1:
        PRFfit = PRFfit + b
    else:
        PRFfit = PRFfit + __polyval2d(bx, by, b)

    # Calculate the sum squared difference between data and model.
    PRFres = np.nansum((DATimg - PRFfit)**2. / DATerr**2.)

    # Keep the fit centered.
    if max(abs(col - x[0]), abs(row - y[0])) > 5.:
        PRFres = 1.e300

    return PRFres


def __prf_to_detector(flux, OBJx, OBJy, DATx, DATy, wx, wy, a,
splineInterpolation):
    '''
    PRF model binned to the detector pixel size.
    '''
    cosa = np.cos(np.radians(a))
    sina = np.sin(np.radians(a))

    # Where in the pixel is the source position?
    PRFfit = np.zeros((np.size(DATy), np.size(DATx)), dtype='float64')

    for i in xrange(len(flux)):
        FRCx, INTx = np.modf(OBJx[i])
        FRCy, INTy = np.modf(OBJy[i])

        if FRCx > 0.5:
            FRCx -= 1.0
            INTx += 1.0

        if FRCy > 0.5:
            FRCy -= 1.0
            INTy += 1.0

        FRCx = -FRCx
        FRCy = -FRCy

        # Constuct model PRF in detector coordinates.
        for (j,y) in enumerate(DATy):
            for (k,x) in enumerate(DATx):
                xx = x - INTx + FRCx
                yy = y - INTy + FRCy
                dx = xx * cosa - yy * sina
                dy = xx * sina + yy * cosa
                PRFfit[j,k] += PRFfit[j,k] + \
                    splineInterpolation(dy * wy, dx * wx) * flux[i]

    return PRFfit


def __polyval(x, c, tensor=True):
    '''
    Evaluate a one-dimensional polynomial at a given point or points.

    Arguments:
    x -- a point or points at which the polynomial will be evaluated
    c -- the coefficients of the polynomial
    tensor -- treat `c` as a tensor
    '''
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)

    if isinstance(x, numpy.ndarray) and tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    c0 = c[-1] * np.ones(x.shape, dtype='float64')

    for i in range(2, c.shape[0] + 1) :
        c0 = c[-i] + c0*x

    return c0


def __polyval2d(x, y, c):
    '''
    Evaluate a 2-dimensional polynomial at a given point or points.

    Arguments:
    x, y -- a point or points at which the polynomial will be evaluated
    c  -- the coefficients of the polynomial
    '''
    c = __polyval(x, c)
    c = __polyval(y, c, tensor=False)

    return c
