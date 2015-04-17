import ephem
thob = ephem.Observer()
thob.long = ephem.degrees("-7.44111")
thob.lat = ephem.degrees("31.9533")
thob.elevation = 1925.0 + 700.0 
thob.date = "2010/1/1" 
jupiter = ephem.Jupiter(thob)
print jupiter.alt, jupiter.az
print jupiter.ra, jupiter.dec
