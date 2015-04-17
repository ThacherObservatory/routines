import sys
import pyfits
import tempfile
import os
import shutil
import glob
import numpy as np


def get_wcs_physical(struct):
    '''
    Gets the physical WCS keywords from `struct`.
    '''
    try:
        crpix1p = struct.header['CRPIX1P']
        crpix2p = struct.header['CRPIX2P']
        crval1p = struct.header['CRVAL1P']
        crval2p = struct.header['CRVAL2P']
        cdelt1p = struct.header['CDELT1P']
        cdelt2p = struct.header['CDELT2P']
    except KeyError:
        raise RuntimeError('Could not extract one or more physical '
            'WCS keywords from the input file.')

    return (crpix1p, crpix2p, crval1p, crval2p, cdelt1p, cdelt2p)


def read_prf_image(infile, hdu):
    '''
    Reads a pixel response file `infile`, indexed by `hdu`.
    '''
    # Open the input file.
    prf = pyfits.open(infile, mode='readonly')

    # Read bitmap image.
    try:
        img = prf[hdu].data
        naxis1 = prf[hdu].header['NAXIS1']
        naxis2 = prf[hdu].header['NAXIS2']
    except KeyError:
        raise RuntimeError('Cannot parse the input file %f.' % infile)
    except IndexError:
        raise RuntimeError('Cannot parse the input file %f.' % infile)

    # Get the physical WCS keywords.
    crpix1p, crpix2p, crval1p, crval2p, cdelt1p, cdelt2p = \
        get_wcs_physical(prf[hdu])

    # Close the input file.
    prf.close()

    return (img, crpix1p, crpix2p, crval1p, crval2p, cdelt1p, cdelt2p)


def read_tpf(infile, colnames):
    '''
    Reads a target pixel data file `infile` and returns header data
    and an array of the columns specified by the list `colnames`.
    '''
    tpf = pyfits.open(infile, mode='readonly', memmap=True)

    # Save all header data.
    try:
        naxis2 = tpf['TARGETTABLES'].header['NAXIS2']
        kepid = tpf[0].header['KEPLERID']
        channel = tpf[0].header['CHANNEL']
        skygroup = tpf[0].header['SKYGROUP']
        module = tpf[0].header['MODULE']
        output = tpf[0].header['OUTPUT']
        quarter = tpf[0].header['QUARTER']
        season = tpf[0].header['SEASON']
        ra = tpf[0].header['RA_OBJ']
        dec = tpf[0].header['DEC_OBJ']

        try:
            kepmag = float(tpf[0].header['KEPMAG'])
        except ValueError:
            kepmag = None

        tdim5 = tpf['TARGETTABLES'].header['TDIM5']
        xdim = int(tdim5.strip().strip('(').strip(')').split(',')[0])
        ydim = int(tdim5.strip().strip('(').strip(')').split(',')[1])

        column = tpf['TARGETTABLES'].header['1CRV5P']
        row = tpf['TARGETTABLES'].header['2CRV5P']
    except KeyError:
        raise RuntimeError('Cannot find one or more header keywords '
            'in the input file %f.' % infile)

    # Read and close TPF data pixel image.
    pixels = [None] * len(colnames)

    for i in xrange(len(colnames)):
        try:
            pixels[i] = np.array(tpf['TARGETTABLES'].data.field(colnames[i]),
                dtype='float64')

            if len(np.shape(pixels[i])) == 3:
                isize, jsize, ksize = np.shape(pixels[i])
                pixels[i] = np.reshape(pixels[i], (isize, jsize*ksize))

        except KeyError:
            print 'Warning: The target pixel file ' + infile + ' does not ' \
                'contain the column ' + colnames[i] + '. It has been ' \
                'ignored.'

    # Close the input file.
    tpf.close()

    return (kepid, channel, skygroup, module, output, quarter, season,
        ra, dec, column, row, kepmag, xdim, ydim, pixels)


def timekeys(instr):
    '''
    Read the time keywords from the FITS structure `instr`.
    '''
    tstart = 0.
    tstop = 0.
    cadence = 0.

    # Extract the BJDREFI field.
    try:
        bjdrefi = instr[1].header['BJDREFI']
    except KeyError:
        bjdrefi = 0.

    # Extract the BJDREFF field.
    try:
        bjdreff = instr[1].header['BJDREFF']
    except KeyError:
        bjdreff = 0.

    bjdref = bjdrefi + bjdreff

    # Extract the TSTART, STARTBJD, or LC_START field as applicable.
    try:
        tstart = instr[1].header['TSTART']
    except KeyError:
        try:
            tstart = instr[1].header['STARTBJD']
            tstart += 2.4e6
        except KeyError:
            try:
                tstart = instr[0].header['LC_START']
                tstart += 2400000.5
            except KeyError:
                try:
                    tstart = instr[1].header['LC_START']
                    tstart += 2400000.5
                except KeyError:
                    raise RuntimeError('Cannot find TSTART, STARTBJD, '
                        'or LC_START in the input file.')

    tstart += bjdref

    # Extract the TSTOP, ENDBJD, or LC_END field as applicable.
    try:
        tstop = instr[1].header['TSTOP']
    except KeyError:
        try:
            tstop = instr[1].header['ENDBJD']
            tstop += 2.4e6
        except KeyError:
            try:
                tstop = instr[0].header['LC_END']
                tstop += 2400000.5
            except KeyError:
                try:
                    tstop = instr[1].header['LC_END']
                    tstop += 2400000.5
                except KeyError:
                    raise RuntimeError('Cannot find TSTOP, STOPBJD, or '
                        'LC_STOP in the input file.')

    tstop += bjdref

    # Extract the OBSMODE or DATATYPE field.
    try:
        cadence = 1.
        obsmode = instr[0].header['OBSMODE']
    except KeyError:
        try:
            obsmode = instr[1].header['DATATYPE']
        except KeyError:
            raise RuntimeError('Cannot find the OBSMODE or DATATYPE '
                'fields in the input file.')

    if 'short' in obsmode:
        cadence = 54.1782
    elif 'long' in obsmode:
        cadence = 1625.35

    return tstart, tstop, bjdref, cadence
