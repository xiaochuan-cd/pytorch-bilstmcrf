import sys
import cv2
import os
from sys import platform
import argparse

try:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        sys.path.append('../../python');
        from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    parser = argparse.ArgumentParser()
    args = parser.parse_known_args()

    params = dict()
    params["model_folder"] = "../../../models/"

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    
    images = os.listdir('/vol/dataset/')
    for i in images:
        datum = op.Datum()
        imageToProcess = cv2.imread('/vol/body/'+i)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])
    
except Exception as e:
    print(e)
    sys.exit(-1)
