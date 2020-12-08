# Final-project

Automatic Identification of
Parallel Growth Among Neuronal
Dendritic Branches

Inbar DAHARI 

Internal advisor: Prof. Boaz BEN MOSHE
External advisor: Prof. Danny BARANES and Dr. Refael MINNES
-Ariel UNIVERSITY-
September 2019
_____________________________________________________________________________________________________________________________________

Installation:
- Drag all .py files to Pycharm software or any other Python debugger.
- You may be prompted to installing certain packages.

The main file to work on is "main_solution_update.py" '
The lines of code in this file that need to be modified in order to update the parameters of each image accordingly are:

1. Row #84: Inside the brackets, write the full address location of the image file. the image must be saved as a png type.
exemple:   C: /Users/inbar/Desktop/result1/10/MOSHE6.png

2. Row #97: in the Canny function you can change the third parameter, threshold1, should be in the range of 100-200.
   dst = cv.Canny(blur, 25, 140, None, 3)  # threshold1= 200- 110- as the num is low- the lines are more detect
        #Python: cv.Canny(image, edges, threshold1, threshold2, aperture_size=3) → None

3. Row #381 + #382: regarding merging lines that are on the same line or with duplicates. you can change the values
#381: min_distance_to_merge = 20 // range can be from 10-40
#382 min_angle_to_merge = 5  // The ideal values can be range from 5-10

** The ideal values were selected after a broad examination of ranges and finding the effective numbers for maximum line marking
