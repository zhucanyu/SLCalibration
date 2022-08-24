# Calibration-Method-for-Structured-Light-System-Based-on-DIC
This is a calibration method for a camera and a speckle projector.  
This is the code mentioned in paper "Calibration Method for Structured Light System Based on DIC",You can get the principle of it in this paper.  
The code is not well organized, so it currently lacks ubiquity and we may improve it in the future.  
You can send an email to "zhucanyu@mail.ustc.edu.cn" if you have any question.
## Tips
The code to calibrate and find circle use OpenCV's findCirclesGrid() ,calibrateCamera() ,stereoCalibrate(). You need the OpenCV library to run or adjust the code. You can use python to use OpenCV to achieve the same function of the program.
## Instruction for use
1.Rename the different positions calibration plate images with speckle as '^.bmp',and plate images without speckle as '^off.bmp', ^ means the numbers from 1 to n,put them in "DIC_code/board_pics". Put the speckle picture in "DIC_code/speckle_pics".  
2.Modify the image format to "png" and put it in one folder. Change the folder name, board size,and Symmetric parameter in the "FindCircles/mian.cpp" code, find the centers and save the coordinates in "centers^.csv" in your folder.  
3.Put these "centers^.csv" in "DIC_code/" , use "DIC_code/main.m" to find matching coordinates and save them as 'centersnew^.csv'.  
4.Put all "1.png", "centers^.csv","centersnew^.csv" and "referencecenters.csv" to one folder,change the folder name in the code, and use "calibrate/main.cpp" to get calibration parameters. The "referencecenters.csv" is the World coordinates of the centers of the calibration plate circle. The calibrated parameters are written to the root in "T.csv","R.csv","distl.csv","distr.csv","mtxl.csv","mtxr.csv".  
5.Put "T.csv","R.csv","distl.csv","distr.csv","mtxl.csv","mtxr.csv" in folder "rebiuld/". You can use open source DIC software (e.g. Ncorr) to get the matching camera image coordinates and scatter coordinates, save them in "cl.csv" and "cr.csv", and use "main.m" to get their 3D coordinates, which would be saved in "pc.csv".
