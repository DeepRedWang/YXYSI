import os

def makefile(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except:
            pass