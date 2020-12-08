# Final-project

Automatic Identification of
Parallel Growth Among Neuronal
Dendritic Branches

Inbar DAHARI 

Ariel UNIVERSITY

Internal advisor: Prof. Boaz BEN MOSHE
External advisor: Prof. Danny BARANES and Dr. Refael MINNES

September 2019

- Drag all files from to Pycharm software or any other Python debugger.
- You may be prompted to download certain directories that are in the files so that you can compile the code. It takes a few minutes.

The file to work on and compile is "main_solution_update.py" '
The lines of code in this file that need to be modified in order to update the parameters of each image accordingly are:

Line No. 77 - Inside the brackets, write the full address of the image and its name, and the image must be saved as a png type.
C: /Users/inbar/Desktop/result1/10/MOSHE6.png
Row # 90- in the function to change the third parameter, threshold1, should be in the range of 100-200
 
 cv.Canny (image, edges, threshold1, threshold2, aperture_size = 3)

Row No. 374 - regarding merging lines that are on the same line or with duplicates.
min_distance_to_merge = 20
The ideal values ​​range from 10-40

Row No. 375 - regarding merging lines that are on the same line or with duplicates.
min_angle_to_merge = 5
The ideal values ​​range from 5-10

* The ideal values ​​for me are after a broad examination of ranges and finding the effective numbers for maximum marking of lines.
