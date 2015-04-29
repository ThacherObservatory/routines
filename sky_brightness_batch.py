import pyfits as pf
import djs_photfrac_mb as pfmb
import pytz,datetime
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy import units as u
import ephem
from astropy import wcs
from astropy.io.fits import open
import robust as rb
from djs_photfrac_mb import *

lat = 34.467028
lon = -119.1773417
deczen = lat

thob = ephem.Observer()
thob.long = ephem.degrees("-119.1773417")
thob.lat = ephem.degrees("34.467028")
thob.elevation = 504.4 

dir = '/Users/jonswift/Dropbox (Thacher)/Observatory/AllSkyCam/Data/13April2015/'
file = 'Image_15_crop2_astrom.fits'

rate = 0.1500
darkrate = 0.0157 # from D. McKenna
mv = -2.5*np.log10(rate - darkrate)+18.865 # from D. McKenna

# Get image and header
image, header = pf.getdata(dir+file, 0, header=True)

# Get image info
date = header["DATE-OBS"]
# From observer log (header time is not right)
time = '00:14:20'
local = pytz.timezone ("America/Los_Angeles")
naive = datetime.datetime.strptime (date+" "+time, "%Y-%m-%d %H:%M:%S")
local_dt = local.localize(naive, is_dst=None)
utc_dt = local_dt.astimezone (pytz.utc)

# All dates and times in pyephem are UTC
thob.date = utc_dt
ra = thob.sidereal_time()
dec = ephem.degrees(np.radians(deczen))


image = np.array(image)
ysz, xsz = np.shape(image)

# Get image astrometry    
hdulist = open(dir+file)
w = wcs.WCS(hdulist['PRIMARY'].header)
radeg  = np.degrees(ra)
decdeg = np.degrees(dec)
xpix,ypix = w.wcs_world2pix(radeg,decdeg,1) # Pixel coordinates of (RA, DEC)

pixsz  = np.sqrt(header['CD1_1']**2 + header['CD1_2']**2)
radpix = 2.5/pixsz 

ap = djs_photfrac(ypix,xpix,radpix,xdimen=xsz,ydimen=ysz)

# Image characteristics and plot
sig = rb.std(image)
med = np.median(image)
vmin = med - 3*sig
vmax = med + 5*sig
plt.figure(1)
plt.clf()
plt.imshow(image,vmin=vmin,vmax=vmax,cmap='gist_heat',interpolation='nearest', \
           origin='lower')
plt.scatter(xpix,ypix,marker='+',s=100,facecolor='none',edgecolor='yellow', \
            linewidth=1.5)
plt.xlim(0,xsz)
plt.ylim(0,ysz)
plt.axis('off')
plt.title('Field Center')
plt.savefig('Center.png',dpi=300)

image2 = image*0
image2[ap['xpixnum'],ap['ypixnum']] = image[ap['xpixnum'],ap['ypixnum']]

plt.figure(2)
plt.clf()
plt.imshow(image2,vmin=vmin,vmax=vmax,cmap='gist_heat',interpolation='nearest', \
           origin='lower')
plt.scatter(xpix,ypix,marker='+',s=100,facecolor='none',edgecolor='yellow', \
            linewidth=1.5)
plt.xlim(0,xsz)
plt.ylim(0,ysz)
plt.axis('off')
plt.title('Photometer Field of View')
plt.savefig('FOV.png',dpi=300)

ftot = np.sum(image[ap['xpixnum'],ap['ypixnum']])
npix = np.sum(ap['fracs'])

avgval = ftot/npix
# Get rid of stars
avgval = np.median(image[ap['xpixnum'],ap['ypixnum']])
avgval = np.min(image[ap['xpixnum'],ap['ypixnum']])


logavg = -2.5*np.log10(avgval)
calval = 21.029 - logavg
fullfile = 'img00015_ds.fit'

# Get image and header
fullim, fullh = pf.getdata(dir+fullfile, 0, header=True)
logim = -2.5*np.log10(fullim)
calim = logim + calval

ysz, xsz = np.shape(calim)
calap = djs_photfrac(ysz/2,xsz/2+40,ysz/2.0,xdimen=ysz,ydimen=xsz)
newcal = calim*np.inf
newcal[calap['xpixnum'],calap['ypixnum']] = calim[calap['xpixnum'],calap['ypixnum']]
cmap='CMRmap_r'
vmin = 20
vmax = 22.5
plt.figure(3)
plt.clf()
plt.imshow(newcal,vmin=vmin,vmax=vmax,cmap=cmap,interpolation='nearest', \
           origin='upper')
plt.axis('off')
plt.colorbar(shrink=0.65,aspect=10,ticks=[20,20.5,21,21.5,22], \
    orientation='horizontal',pad=0.075)
plt.annotate('N',[0.52,0.91],horizontalalignment='center',xycoords='figure fraction',\
    fontsize=18)
plt.annotate('S',[0.52,0.23],horizontalalignment='center',xycoords='figure fraction',\
    fontsize=18)
plt.annotate('E',[0.25,0.58],horizontalalignment='center',xycoords='figure fraction',\
    fontsize=18)
plt.annotate('W',[0.795,0.58],horizontalalignment='center',xycoords='figure fraction',\
    fontsize=18)
plt.annotate(r'mags/arcsec$^2$',[0.5,0.07],horizontalalignment='center', \
             xycoords='figure fraction',fontsize=12)
plt.savefig('SkyBrightness.png',dpi=300)