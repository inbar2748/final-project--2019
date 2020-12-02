
"""
@final project 2019 by Inbar DAHARI
"""



#
#  Copyright (c) 2019  INBAR DAHARI.
#  All rights reserved.
#

import sys
import math
from typing import List

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import PyQt5
import seaborn as sns
import matplotlib.cm as cm

from Distance import DistanceBL

from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FormatStrFormatter


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "(x: {0}, y: {1})".format(self.x, self.y)

    def tuple(self):
        return print(self.x, self.y)

class Dendrite:

    def __init__(self, id, length, vector1, vector2, angle):
        self.id = id
        self.length = length
        self.vector1 = vector1
        self.vector2 = vector2
        self.angle = angle

    def __eq__(self, other):
        return self.angle == other.angle

    def __lt__(self, other):
        return self.angle < other.angle

    def __str__(self):
        return "id: {0} , length: {1}, vector1: {2}, vector2: {3}, angle: {4} ".format(self.id, self.length, self.vector1, self.vector2, self.angle)

class Range:

    def __init__(self, min, max, id, angle, length, dendrite):
        self.min = min
        self.max = max
        self.id= id
        self.angle= angle
        self.length = length
        self.dendrite = dendrite

    def __str__(self):
        return "Dendrite: {0} , \nrange: {1}---> {2}".format(self.dendrite, self.min, self.max)


class interface:

    def get_lines(self, lines_in):
        if cv.__version__ < '3.0':
            return lines_in[0]
        return [l[0] for l in lines_in]

    def main(self, argv):

        default_file = 'den.png'
        filename = argv[0] if len(argv) > 0 else default_file
        # Loads an image
        src = cv.imread('C:/Users/inbar/Desktop/E56a 9 RGB.png', cv.IMREAD_GRAYSCALE)

        # Check if image is loaded fine
        if src is None:
            print ('Error opening image!')
            print ('Usage: hough_lines.py [' + default_file + '] \n')
            return -1
        """ 
        img = cv.imread('_.png')
        imgplot = plt.imshow(img)
        plt.show(imgplot)
        """
        blur = cv.GaussianBlur(src, (5, 5), 0)
        dst = cv.Canny(blur, 25, 140, None, 3)  # threshold1= 200- 110- as the num is low- the lines are more detect
        #Python: cv.Canny(image, edges, threshold1, threshold2, aperture_size=3) → None
        # threshold1 – first threshold for the hysteresis procedure
        cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR) # turn to binary img
        """
            with the arguments:
            dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
            lines: A vector that will store the parameters ( x start ,y start,x end,y end) of the detected lines
            rho : The resolution of the parameter r in pixels. We use 1 pixel.
            theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
            threshold: The minimum number of intersections to "*detect*" a line
            minLinLength: The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
            maxLineGap: The maximum gap between two points to be considered in the same line.
            """
        lines = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
        # It gives as output the extremes of the detected lines (x0,y0,x1,y1)

        # prepare
        _lines = []
        for _line in self.get_lines(lines):
            _lines.append([(_line[0], _line[1]), (_line[2], _line[3])])

        # sort
        _lines_x = []
        _lines_y = []
        for line_i in _lines:
            orientation_i = math.atan2((line_i[0][1] - line_i[1][1]), (line_i[0][0] - line_i[1][0]))
            if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90 + 45):
            #if ((orientation_i)*(180*math.pi) > 45) and ((orientation_i)*(180*math.pi)< (90 + 45)):
                _lines_y.append(line_i)
            else:
                _lines_x.append(line_i)
        #sort the lines by the beggining  point- x and y
        _lines_x = sorted(_lines_x, key=lambda _line: _line[0][0])
        _lines_y = sorted(_lines_y, key=lambda _line: _line[0][1])

        merged_lines_x = self.merge_lines_pipeline_2(_lines_x)
        merged_lines_y = self.merge_lines_pipeline_2(_lines_y)

        #list of all the merged lines
        merged_lines_all = []
        merged_lines_all.extend(merged_lines_x)
        merged_lines_all.extend(merged_lines_y)
        print("process groups lines", len(_lines), len(merged_lines_all),'\n')
        img_merged_lines = cdst

        DendriteList = []
        id = 0
        RangeMap = dict()
        for line in merged_lines_all:
            cv.line(img_merged_lines, (line[0][0], line[0][1]), (line[1][0], line[1][1]), (0, 0, 255), 3, cv.LINE_AA)

            # length of each line
            x0 = line[0][0]
            x1 = line[1][0]
            y0 = line[0][1]
            y1 = line[1][1]
            dist = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            #print('The length : ', dist)

            # finding the angle between start point and end point of line
            v1 = Vector(x0, y0)  # coordinate of the lines
            v2 = Vector(x1, y1)

            # the coordinate of the marks lines- Each line is represented by four numbers, which are the two endpoints of the detected line segment
            #print(merged_lines_all[line])
            #Vector.tuple(v1)
            #Vector.tuple(v2)
            #_______________
            #__________________________________________
            radians = math.atan2((line[0][1] - line[1][1]), (line[0][0] - line[1][0])) #in radians
            turn_degrees = math.degrees(radians)

            #print(orientation_i)
            #orientation_i = math.atan2((line[0][1] - line[1][1]), (line[0][0] - line[1][0]))

            #in Radians(180 / math.pi) or in degrees(180 * math.pi)

            #print('angle:', radians, '\n')
            id = id + 1
            DendriteList.append(Dendrite(id, dist, v1, v2,turn_degrees%180))#%360

        print("information for each dendrite: ")
        DendriteList.sort()


        x= ( 180 / len(DendriteList) )#angle to group
        for dendrite in DendriteList:
            min = dendrite.angle - x
            max = dendrite.angle + x
            range = Range (min, max, dendrite.id, dendrite.angle, dendrite.length, dendrite)
            if RangeMap.get(range) == None:
                RangeMap.update({range: []})
            print (dendrite)
        print ('\n',"<--------------- classification of parallel groups: --------------->", '\n')

        for dendrite in DendriteList:
            angle_another = dendrite.angle
            id_another= dendrite.id
            for key,value in RangeMap.items():
                if(angle_another >= key.min and angle_another <=  key.max and id_another!=key.id):
                    value.append(dendrite)
        ModifyRangeMap = dict()
        NotParallelMap = dict()
        for key, value in RangeMap.items():
            if len(value) != 0:
                ModifyRangeMap.update({key:value})
            else:
                NotParallelMap.update({key:value})


        print ("Parallel:\n")
        for key, value in ModifyRangeMap.items():
            print('\nKey: {0}: \nnumber of parallel lines: {1}'.format(key, len(value)), *value, sep='\n')

        print("\nNot Parallel:\n")
        for key, value in NotParallelMap.items():
            print('{0}: {1} number of parallel lines: {2}\n'.format(key, value, len(value)))


        #----------------short distance between lines------------------

        ComputeDistance = DistanceBL()
        DistanceComputed = dict()
        for key, value in ModifyRangeMap.items():
            a1 = np.array([key.dendrite.vector1.x, key.dendrite.vector1.y, 1])
            a0 = np.array([key.dendrite.vector2.x, key.dendrite.vector2.y, 1])
            shortlist = []
            for parallel_dendirte in value:
                b0 = np.array([parallel_dendirte.vector1.x, parallel_dendirte.vector1.y, 1])
                b1 = np.array([parallel_dendirte.vector2.x, parallel_dendirte.vector2.y, 1])
                temp = ComputeDistance.closestDistanceBetweenLines(a0, a1, b0, b1)
                shortlist.append(temp)
            DistanceComputed.update({key.id:shortlist})

        print('\n The shortest distance batween parallel groups: \n')
        for key, value in DistanceComputed.items():
            print('{0}: {1} \n'.format(key, value))

        #----------------average of all the parallels lines:------------------

        def average(l):
            sumLength=0
            for key, value in l.items():
                sumLength = sumLength + key.length
            total = sumLength
            total = float(total)
            return total/len(l)


        print('\nAverage of all the parallels lines:', (average(ModifyRangeMap)))

        print('\nAverage of all the NOT parallels lines:', (average(NotParallelMap)))



        #----------------------------------------------------------------------

        cv.imwrite('prediction/merged_lines.jpg', img_merged_lines)
        plt.subplot(121), plt.imshow(src), plt.title('original after blurring')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img_merged_lines), plt.title('Detected Lines - Probabilistic Line Transform')
        plt.xticks([]), plt.yticks([])

        plt.show()

        # ------------fig1 -angles scatter plot of dendrites------
        x1 = [0] * (len(DendriteList) + 1)
        y1 = [0] * (len(DendriteList) + 1)

        i = 0
        for dendrite in DendriteList:
            # x-axis values
            xAngle = dendrite.angle
            x1[i] = xAngle

            # y-axis values
            yId = dendrite.id
            y1[i] = yId

            i = i + 1

        # Get the color for each sample.
        colors = cm.rainbow(np.linspace(0, 1, len(y1)))

        # plot the data
        fig= plt.subplots()
        ax = plt.subplot(2, 1, 1)
        ax.set_xlim(0, 180)
        ax.set_ylim(0, len(y1))
        ax.scatter(x1,y1, color=colors)
        ax.set(xlabel="x - Angles", ylabel="y - ID")
        ax.set(title="angles")

        # tell matplotlib to use the format specified above
        ax.xaxis.set_major_locator(MultipleLocator(180 / len(DendriteList)))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.grid(True, which='minor')
        ax.xaxis.set_minor_locator(MultipleLocator(180 / len(DendriteList)))
        ax.tick_params(axis='x', rotation=70)

        plt.title('figure 1- Range of angles:')


        #___________
        # _______________________________

        plt.subplot(2, 1, 2)
        plt.hist( x1, bins=int(180/(180/len(DendriteList))) ,range=[0, 180],rwidth=1, color='b', edgecolor='black', lw=0)
        ax = plt.gca()
        ax.set_xlim(0, 180)
        ax.xaxis.set_major_locator(MultipleLocator(180 / len(DendriteList)))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(180 / len(DendriteList)))
        ax.tick_params(axis='x', rotation=70)
        ax.set(xlabel="x - Angles", ylabel="# of dendrite")
        plt.grid(True)
        plt.show()

        # ------------fig2 -length scatter plot of dendrites------
        sns.set(color_codes=True)
        x2 = [0] * (len(ModifyRangeMap)+1 )
        x3 = [0] * (len(NotParallelMap)+1 )

        i = 0
        for key, value in ModifyRangeMap.items():
            # x-axis values
            xLength = key.length
            x2[i] = xLength
            i = i + 1
        j=0
        for key, value in NotParallelMap.items():
            # x-axis values
            x3Length = key.length
            x3[j] = x3Length
            j = j + 1

        x2.sort()
        x3.sort()

        plt.subplot(2, 1, 1)
        plt.hist(x3, bins=int(180 / (180 / len(DendriteList))), range=[0, x3[len(x3) - 1]], rwidth=1, color='blue',edgecolor='black', lw=1)
        ax = plt.gca()
        ax.set_ylim(0, len(x3))
        ax.set_xlim(0, x3[len(x3) - 1])
        ax.xaxis.set_major_locator(MultipleLocator(180 / len(DendriteList)))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(180 / len(DendriteList)))
        ax.tick_params(axis='x', rotation=80)
        ax.set(xlabel="x - length", ylabel="# of dendrites")
        ax.set_facecolor('#d8dcd6')
        plt.suptitle('figure 2- Range of length:', fontsize=14, fontweight='bold')
        plt.title('NOT parallel groups vs. parallel groups')
        plt.grid(True)


        plt.subplot(2, 1, 2)
        plt.hist(x2, bins=int(180 / (180 / len(DendriteList))), range=[0, x2[len(x2)-1]], rwidth=1, color='blue', edgecolor='black',lw=1)
        ax = plt.gca()
        ax.set_xlim(0, x2[len(x2)-1])
        ax.xaxis.set_major_locator(MultipleLocator(180 / len(DendriteList)))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis='x', rotation=80)
        ax.set(xlabel="x - length", ylabel="# of dendrites")
        ax.set_facecolor('#d8dcd6')
        plt.grid(True)
        plt.show()
        # ------------------
        # ------------fig3 -length scatter plot of dendrites------
       
        # ------------------


        return merged_lines_all



    def merge_lines_pipeline_2(self, lines):
        super_lines_final = []
        super_lines = []
        min_distance_to_merge = 20 #30
        min_angle_to_merge = 5 #30

        for line in lines:
            create_new_group = True
            group_updated = False

            for group in super_lines:
                for line2 in group:
                    if self.get_distance(line2, line) < min_distance_to_merge:
                        # check the angle between lines
                        orientation_i = math.atan2((line[0][1] - line[1][1]), (line[0][0] - line[1][0]))
                        orientation_j = math.atan2((line2[0][1] - line2[1][1]), (line2[0][0] - line2[1][0]))

                        if int(abs(
                                abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge:
                            #print("angles", orientation_i, orientation_j)
                            #print(int(abs(orientation_i - orientation_j)))
                            group.append(line)

                            create_new_group = False
                            group_updated = True
                            break

                if group_updated:
                    break

            if (create_new_group):
                new_group = []
                new_group.append(line)

                for idx, line2 in enumerate(lines):
                    # check the distance between lines
                    if self.get_distance(line2, line) < min_distance_to_merge:
                        # check the angle between lines -finding the angle between start point and end point of each line
                        orientation_i = math.atan2((line[0][1] - line[1][1]), (line[0][0] - line[1][0]))
                        orientation_j = math.atan2((line2[0][1] - line2[1][1]), (line2[0][0] - line2[1][0]))

                        if int(abs(abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge:
                            #print("angles", orientation_i, orientation_j)
                            #print(int(abs(orientation_i - orientation_j))) #אthe differenceof the angle beetwin two lines

                            new_group.append(line2)

                            # remove line from lines list
                            # lines[idx] = False
                # append new group
                super_lines.append(new_group)

        for group in super_lines:
            super_lines_final.append(self.merge_lines_segments1(group))

        return super_lines_final


    def merge_lines_segments1(self, lines, use_log=False):
        #if there is just one line
        if (len(lines) == 1):
            return lines[0]

        line_i = lines[0]

        # orientation
        orientation_i = math.atan2((line_i[0][1] - line_i[1][1]), (line_i[0][0] - line_i[1][0]))
        #print(orientation_i)


        points = []
        for line in lines:
            points.append(line[0])
            points.append(line[1])

        #if ((orientation_i)*(180*math.pi) > 45) and ((orientation_i)*(180*math.pi)< (90 + 45)):
        if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90 + 45):

            # sort by y
            points = sorted(points, key=lambda point: point[1])

            if use_log:
                print("use y")
        else:

            # sort by x
            points = sorted(points, key=lambda point: point[0])

            if use_log:
                print("use x")

        return [points[0], points[len(points) - 1]]

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
        # https://stackoverflow.com/questions/32702075/what-would-be-the-fastest-way-to-find-the-maximum-of-all-possible-distances-betw


    def lines_close(self, line1, line2):
        dist1 = math.hypot(line1[0][0] - line2[0][0], line1[0][0] - line2[0][1])
        dist2 = math.hypot(line1[0][2] - line2[0][0], line1[0][3] - line2[0][1])
        dist3 = math.hypot(line1[0][0] - line2[0][2], line1[0][0] - line2[0][3])
        dist4 = math.hypot(line1[0][2] - line2[0][2], line1[0][3] - line2[0][3])

        if (min(dist1, dist2, dist3, dist4) < 100):
            return True
        else:
            return False

    #
    def lineMagnitude(self, x1, y1, x2, y2):
        lineMagnitude = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return lineMagnitude

        # Calc minimum distance from a point and a line segment (i.e. consecutive vertices in a polyline).
        # https://nodedangles.wordpress.com/2010/05/16/measuring-distance-from-a-point-to-a-line-segment/
        # http://paulbourke.net/geometry/pointlineplane/


    def DistancePointLine(self,px, py, x1, y1, x2, y2):
        # http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
        LineMag = self.lineMagnitude(x1, y1, x2, y2)

        if LineMag < 0.00000001:
            DistancePointLine = 9999
            return DistancePointLine

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (LineMag * LineMag)

        if (u < 0.00001) or (u > 1):
            # // closest point does not fall within the line segment, take the shorter distance
            # // to an endpoint
            ix = self.lineMagnitude(px, py, x1, y1)
            iy = self.lineMagnitude(px, py, x2, y2)
            if ix > iy:
                DistancePointLine = iy
            else:
                DistancePointLine = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            DistancePointLine = self.lineMagnitude(px, py, ix, iy)

        return DistancePointLine


    def get_distance(self,line1, line2):
        dist1 = self.DistancePointLine(line1[0][0], line1[0][1],line2[0][0], line2[0][1], line2[1][0], line2[1][1])
        dist2 = self.DistancePointLine(line1[1][0], line1[1][1],line2[0][0], line2[0][1], line2[1][0], line2[1][1])
        dist3 = self.DistancePointLine(line2[0][0], line2[0][1],line1[0][0], line1[0][1], line1[1][0], line1[1][1])
        dist4 = self.DistancePointLine(line2[1][0], line2[1][1],line1[0][0], line1[0][1], line1[1][0], line1[1][1])

        return min(dist1, dist2, dist3, dist4)



if __name__ == "__main__":
    Main = interface()
    Main.main(sys.argv[1:])


