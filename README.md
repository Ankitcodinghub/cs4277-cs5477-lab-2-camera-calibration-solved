# cs4277-cs5477-lab-2-camera-calibration-solved
**TO GET THIS SOLUTION VISIT:** [CS4277/CS5477 Lab 2-Camera Calibration Solved](https://www.ankitcodinghub.com/product/cs4277-cs5477-lab-2-camera-calibration-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;94845&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CS4277\/CS5477 Lab 2-Camera Calibration Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column">
Camera Calibration

[1]:

</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
<pre>import cv2
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import lab2
%matplotlib inline
</pre>
</div>
</div>
</div>
<div class="layoutArea">
<div class="column">
1.0.1 Introduction

In this assignment, you will implement Zhenyou Zhang’s camera calibration. The extrincs and in- trincs of a camera are estimated from three images of a model plane. You will first estimate the five intrinsic parameters (focal length, principle point, skew) and six extrinsic parameters (three for ro- tation and three for translation) by a close-form solution. Then you will estimate five distortion parameters and also finetune all parameters by minimize the total reprojection error.

This assignment is worth 15% of the final grade.

References:

* Lecture 4

* Zhengyou Zhang. A Flexible New Technique for CameraCalibration

1.0.2 Instructions

This workbook provides the instructions for the assignment, and facilitates the running of your code and visualization of the results. For each part of the assignment, you are required to complete the implementations of certain functions in the accompanying python file (lab2.py).

To facilitate implementation and grading, all your work is to be done in that file, and you only have to submit the .py file.

Please note the following:

1. Fill in your name, email, and NUSNET ID at the top of the python file. 2. The parts you need to implement are clearly marked with the following: “‘

“”” YOUR CODE STARTS HERE “””

“”” YOUR CODE ENDS HERE “””

“‘

<pre>and you should write your code in between the above two lines.
</pre>
3. Note that for each part, there may certain functions that are prohibited to be used. It is important NOT to use those prohibited functions (or other functions with similar func- tionality). If you are unsure whether a particular function is allowed, feel free to ask any of the TAs.

1.0.3 Submission Instructions

Upload your completed lab2.py onto the relevant work bin in Luminus.

</div>
</div>
<div class="layoutArea">
<div class="column">
1

</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="layoutArea">
<div class="column">
[2]:

</div>
</div>
<div class="layoutArea">
<div class="column">
1.1 Part 1: Load and Visualize Data

In this part, you will get yourself familiar with the data by visualizing it. The data includes three images of a planar checkerboard (CalibIm1-3.tif) and the correpsonding corner locations in each image (data1-3.txt). The 3D points of the model are stored in Model.txt. Note that only X and Y coordinates are provided because we assume that the model plane is on Z = 0. You can visualize the data with the provided code below.

</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
<pre>Model = np.loadtxt('./zhang_data/Model.txt')
X = Model[:, 0::2].reshape([1, -1])
Y = Model[:, 1::2].reshape([1, -1])
pts_model = np.vstack([X, Y])
</pre>
pts_model_homo = np.concatenate([pts_model, np.ones([1, pts_model.shape[1]])],␣ 􏰅→axis= 0)

<pre>pts_2d = []
for i in range(3):
</pre>
<pre>    data = np.loadtxt('./zhang_data/data{}.txt'.format(i+1))
    img = cv2.imread('./zhang_data/CalibIm{}.tif'.format(i+1))
    x = data[:, 0::2].reshape([1, -1])
    y = data[:, 1::2].reshape([1, -1])
    pts_2d.append(np.vstack([x, y]))
</pre>
<pre>    # Visualize images and the corresponding corner locations.
</pre>
<pre>    for j in range(x.shape[1]):
        cv2.circle(img, (np.int32(x[0, j]), np.int32(y[0, j])) , 5, (255, 0, 0),␣
</pre>
􏰅→2)

plt.figure() plt.imshow(img)

</div>
</div>
</div>
<div class="layoutArea">
<div class="column">
2

</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="layoutArea">
<div class="column">
3

</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="layoutArea">
<div class="column">
1.2 Part 2: Estimate the Intrinsic Parameters

In this part, you will estimate the the intrinsics, which inludes focal length, skew and principle point.You will firstly estimate the homography between each observed image and the 3D model. Note that you are allowed to use cv2.findHomography() here to since you already implemented it in lab1. Each view of the checkerboard gives us two constraints:

vb = 0,

where v is 2 × 6 matrix made up of the homography terms. Given three observations, we can get : Vb = 0,

where V is a 6 × 6 matrix obtained from stacking all constraints together. The solution can be obtained by taking the right null-space of V, which is the right singular vector corresponding to the smallest singular value of V.

Implement the following function(s): cv2.calibrateCamera()

• You may use the following functions: cv2.findHomography(), np.linalg.svd()

• Prohibited Functions: cv2.calibrateCamera()

1.3 Part 3: Estimate the Extrinsic Parameters

In this part, you will estimate the extrinsic parameters based on the intrinsic matrix A you ob-

tained from Part 2. You can compute the rotation and translation according to: r1 = λA−1h1r2 = λA−1h2r3 = r1 × r2t = λA−1h3.

λ = 1/∥A−1h1∥ = 1/∥A−1h2∥, and hi represents the ith column of the homography H. Note that the rotation matrix R = [r1, r1, r1] does not in general satisfy the properties of a rotation matrix. Hence, you will use the provided function convt2rotation() to estimate the best rotation matrix. The detail is given in the supplementary of the reference paper.

• You may use the following functions:

<pre>     np.linalg.svd(), np.linalg.inv(),np.linalg.norm(), convt2rotation
</pre>
</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
<pre>R_all, T_all, K = init_param(pts_model, pts_2d)
A = np.array([K[0], K[1], K[2], 0, K[3], K[4], 0, 0, 1]).reshape([3, 3])
img_all = []
for i in range(len(R_all)):
</pre>
<pre>    R = R_all[i]
    T = T_all[i]
    points_2d = pts_2d[i]
    trans = np.array([R[:, 0], R[:, 1], T]).T
    points_rep = np.dot(A, np.dot(trans, pts_model_homo))
    points_rep = points_rep[0:2] / points_rep[2:3]
    img = cv2.imread('./zhang_data/CalibIm{}.tif'.format(i + 1))
    for j in range(points_rep.shape[1]):
</pre>
</div>
</div>
</div>
<div class="layoutArea">
<div class="column">
[3]:

</div>
</div>
<div class="layoutArea">
<div class="column">
4

</div>
</div>
</div>
<div class="page" title="Page 5">
<div class="section">
<div class="layoutArea">
<div class="column">
cv2.circle(img, (np.int32(points_rep[0, j]), np.int32(points_rep[1,␣ 􏰅→j])), 5, (0, 0, 255), 2)

cv2.circle(img, (np.int32(points_2d[0, j]), np.int32(points_2d[1, j])),␣ 􏰅→4, (255, 0, 0), 2)

<pre>   plt.figure()
   plt.imshow(img)
</pre>
</div>
</div>
</div>
<div class="layoutArea">
<div class="column">
Up to now, you already get a rough estimation of the intrinsic and extrinsic parameters. You can check your results with the provided code, which visualizes the reprojections of the corner locations with the estimated parameters. You will find that the points that are far from the center of the image (the four corners of the checkerboard) are not as accurate as points at the center. This is because we did not consider the distortion parameters in this step.

</div>
</div>
<div class="layoutArea">
<div class="column">
5

</div>
</div>
</div>
<div class="page" title="Page 6">
<div class="layoutArea">
<div class="column">
6

</div>
</div>
</div>
<div class="page" title="Page 7">
<div class="layoutArea">
<div class="column">
1.4 Part 4: Estimate All Parameters

In this part, you will estimate all parameters by minimize the total reprojection error:

nm

argmin ∑ ∑∥xij − π(K, R, t, ˇ, Xj)∥.

K,R,t,ˇ i=1 j=1

K, R, t are the intrinsics and extrinsices, which are initialized with estimation from Part 3. ˇ rep- resents the five distortion parameters and are initialized with zeros. Xj and xij represent the 3D model and the corresponding 2D observation.

Note that you will use the function least_squares() in scipy to minimize the reprojec- tion error and find the optimal parameters. During the optimization process, the rotation matrix R should be represented by a 3-dimensional vector by using the provided function matrix2vector(). We provide the skeleton code of how to use the function least_squares() below.

The key step of the optimization is to define the error function error_fun(), where the first parameter param is the parameters you will optimize over. The param in this example includes: intrinsics (0-5), distortion (5-10), extrinsics (10-28). The extrincs consist of three pairs of rotation s and translation t because we have three views. The rotation s is the 3-dimensional vector represen- tation, which you can convert back to a rotation matrix with provided function vector2matrix().

You will have to consider the distortion when computing the reprojection error. Let x = (x, y) be the normalized image coordinate, namely the points_ud_all in the code. The radial distortion is given by:

􏰃xr􏰄 2 4 6 􏰃x􏰄 xr=y =(1+κ1r+κ2r+κ5r)y,

r

where r2 = x2 + y2 and κ1, κ2, κ5 are the radial distortion parameters. The tangential distortion is

given by :

􏰃2κ3xy + κ4(r2 + 2×2)􏰄 dx= κ3(r2+2y2)+2κ4xy ,

where κ3,κ4 are the tangential distortion parameters. FInally, the image coordinates after distor- tion is given by :

xd = xr + dx.

The optimization converges when the error does not change too much. Note that you will de- cide the iter_num according to the error value by yourself. You can verify the optimal parameters by visualizing the points after distortion. The function visualize_distorted() is an example of how to visualize the the points after distortion in image. You will find that the points that are far from the center of the image is more accurate than the estimation from Part 3.

[4]:

</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
<pre>iter_num = 20
param = []
param.extend(K)
k = np.zeros([5,])
param.extend(k)
for i in range(len(R_all)):
</pre>
<pre>    S = matrix2vector(R_all[i])
    param.extend(S)
    param.extend(T_all[i])
</pre>
<pre>param = np.array(param)
</pre>
</div>
</div>
</div>
<div class="layoutArea">
<div class="column">
7

</div>
</div>
</div>
<div class="page" title="Page 8">
<div class="section">
<div class="layoutArea">
<div class="column">
<pre>for i in range(iter_num):
    opt = least_squares(error_fun, param, args=(pts_model, pts_2d))
    param = opt.x
    error = opt.cost
</pre>
<pre>#     print('iteration:', i, 'error:', error)
</pre>
<pre>visualize_distorted(param, pts_model, pts_2d)
</pre>
</div>
</div>
</div>
<div class="layoutArea">
<div class="column">
8

</div>
</div>
</div>
<div class="page" title="Page 9">
<div class="layoutArea">
<div class="column">
9

</div>
</div>
</div>
