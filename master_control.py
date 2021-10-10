import os
import time

counter = 1

time.sleep(10)

os.system("python3 /home/pi/mm_control2.py -hm")
time.sleep(2)
while counter < 5:
     os.system("python3 /home/pi/mm_control2.py -d 200")
     counter += 1
     time.sleep(2)
time.sleep(10)
#os.system("sudo shutdown -h now")
