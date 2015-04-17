import pyfits
import os
import pdb
import string
import pprint as pp
import sys
from select import select





helpstring="""Usage \n python perform_astrometric_calibration [data_dir] [file list separated by line breaks]"""



astrometrydotnet_dir="/usr/local/astrometry/"
data_dir="/Users/jonswift/Astronomy/Caltech/MINERVA/Observing/data/2013Nov24/"
filelist=data_dir+"for_astrometry.txt"
filestoconsider=os.listdir(data_dir)
pp.pprint(filestoconsider)

with open(filelist) as targetlist:
        for filename in targetlist.readlines():
print "Press any key to quit, continuing in 1 second..."
timeout=1
rlist, wlist, xlist = select([sys.stdin], [], [], timeout)
if rlist:
        break

print filename
filepath=os.path.join(data_dir,filename)
try:
        #see if the file already has ra, dec in header
data, header = pyfits.getdata(filepath, 0,
                              header=True)
                
#convert
h,m,s = header['OBJCTRA'].split()
RAdeg = float(h)*15+float(m)*15.0/60+float(s)*15.0/3600
d,m,s = header['OBJCTDEC'].split()
if float(d)<0:
        DECdeg = float(d)-float(m)/60-float(s)/3600
                if float(d)>0:
                    DECdeg = float(d)+float(m)/60+float(s)/3600
                print "\n\nFound RA and DEC info in header\n\n"

                #data, header = pyfits.getdata(filepath, 0, header=True)
                #pdb.set_trace()
                outputdir=os.path.join(data_dir, 'result_'+filename)
                command=string.join(
                        [astrometrydotnet_dir+"/bin/./solve-field",
                        filepath.rstrip(),
                        "--scale-units arcsecperpix --scale-low 0.05 --scale-high 1.5",
                        " --no-fits2fits ", 
                        "--ra ",str(RAdeg)," --dec ", str(DECdeg), 
                        "--radius 1",
                        "--downsample 4", 
                        "--no-plots",
                        "--skip-solved",
                        "--objs 30",
                        "--odds-to-tune-up 1e4",
                        "--dir",outputdir.rstrip(),"--overwrite"]) 
                os.system(command)        
            
            except:
####
                #data, header = pyfits.getdata(filepath, 0, header=True)
                #pdb.set_trace()
outputdir=os.path.join(data_dir, "solved")
command=string.join(
        [astrometrydotnet_dir+"/bin/./solve-field",
         filepath.rstrip(),
         "--scale-units arcsecperpix --scale-low 0.05 --scale-high 1.5",
         " --no-fits2fits ", 
         "--downsample 4", 
         "--no-plots",
         "--skip-solved",
         "--objs 30",
         "--odds-to-tune-up 1e4",
         "--dir",outputdir.rstrip(),"--overwrite"]) 

os.system(command)        

move = "mv "+outputdir+"/solved.new "+data_dir+"/"+filename
os.system(command)
