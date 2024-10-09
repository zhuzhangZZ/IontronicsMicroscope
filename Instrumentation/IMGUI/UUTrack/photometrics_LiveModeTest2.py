# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 14:11:46 2023

@author: iSCAT
"""

import time
import cv2
import numpy as np
import sys,os
photometricsDriverPath = 'C:\ZhuZhang\Iontronics4BSI\IontronicsMicroscope\Instrumentation\IMGUI\\UUTrack\Controller\devices\photometrics\src'

sys.path.append(photometricsDriverPath)
from pyvcam import pvc
from pyvcam.camera import Camera

def main():
    pvc.init_pvcam()
    cam = next(Camera.detect_camera())
    cam.open()
    cam.start_live(exp_time=1)
    cnt = 0
    tot = 0
    t1 = time.time()
    start = time.time()
    width = 100
    height = 200
    dim = (width, height)
    fps = 0
    cam.set_roi(0, 0, width, height)
    while True:
        frame, fps, frame_count = cam.poll_frame()
        #frame['pixel_data'] = cv2.resize(frame['pixel_data'], dim, interpolation = cv2.INTER_AREA)
        frame['pixel_data'] = frame['pixel_data']
        cv2.imshow('Live Mode', frame['pixel_data'])

        low = np.amin(frame['pixel_data'])
        high = np.amax(frame['pixel_data'])
        average = np.average(frame['pixel_data'])

        if cnt == 10:
                t1 = time.time() - t1
                fps = 10/t1
                t1 = time.time()
                cnt = 0
        if cv2.waitKey(10) == 27:
            break
        print('Min:{}\tMax:{}\tAverage:{:.0f}\tFrame Rate: {:.1f}\n'.format(low, high, average, fps))
        cnt += 1
        tot += 1

    cam.finish()
    cam.close()
    pvc.uninit_pvcam()

    print('Total frames: {}\nAverage fps: {}\n'.format(tot, (tot/(time.time()-start))))
if __name__ == "__main__":
    main()