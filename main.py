import viddec as vd
from helmetdetection import helmetimg as himg


vd.bikemode = True
vd.detect_vehicle()
#time.sleep(40)
vd.bikemode = False

