# coding:utf-8
import os, sys
import subprocess
import time
from datetime import timedelta, datetime
import ffmpeg

#infile = sys.argv[1]
#input as parameter
infile = 'TLC00007.AVI'

framerate = 15

s = os.stat(infile)
# get the creation datetime and convert it to a string
created = s.st_birthtime
created_tuple = time.localtime(created)
# created = datetime(*created_tuple[:7])
created = datetime.fromtimestamp(created)
print(created)
created_str = time.strftime("%Y%m%d%H%M", created_tuple)

# create a new folder with the creation datetime
outdir = infile.replace('.AVI', '_' + created_str)
# if the folder exists we assume this .avi 
# file was already porcessed, so we skip it
try:
    os.mkdir(outdir)
except FileExistsError:
    sys.exit()

#invoke ffmpeg
# command = 'ffmpeg -skip_frame nokey -i %s -vsync 0 -r 30 -f image2 %s/%%d.jpeg' % (
command = 'ffmpeg -skip_frame nokey -i %s -f image2 -q:v 1 %s/%%05d.jpeg' % (
    infile, outdir
)
print(command)


# ####

img_files = os.listdir(outdir)
img_files = [x for x in img_files if x.endswith('.jpeg')]
img_files = [x for x in img_files if not x.startswith('._')]

for img_f in sorted(img_files):
    n_str = img_f.replace('.jpeg', '')
    # print("%s %s" % (img_f, n_str))
    n = int(n_str)

    taken = created + timedelta(seconds=framerate * n)
    # taken_tuple = time.localtime(taken)
    taken_str = taken.strftime("%Y-%m-%d_%H-%M-%S")

    # print("%s %5d %s" % (img_f, n, taken_str))
    os.rename(
        os.path.join(outdir, img_f),
        os.path.join(outdir, taken_str + '.jpeg'),
    )

