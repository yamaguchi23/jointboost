JointBoost
==================

This is a C++ implementation of JointBoost [1].

[1] A. Torralba, K. P. Murphy and W. T. Freeman, "Sharing visual features for multiclass and multiview object detection," PAMI, 2007.

Build
------------------

<h5>Requirement</h5>  
- CMake ([http://www.cmake.org/](http://www.cmake.org/))

1) >cmake .  
2) >make

Data format
------------------

The format of training and test data file:

    <label> <feature>:<value> <feature>:<value> ... <feature>:<value>  
       .  
       .  
       .  

    <label> = {0, 1,...,n-1}, n: the number of classes  
    <feature>: feature index (integer value starting from 1)  
    <value>: feature value (double)'

You can use pendigits data as a sample data set.
([http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#pendigits](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#pendigits))

Usage
------------------

<h5>Training</h5>  
    >./jointtrain [options] training_set_file [model_file]  
    options:  
      -r: the number of rounds [default:100]  
      -v: verbose'

<h5>Prediction</h5>  
    >./jointpredict [options] test_set_file model_file  
     options:  
       -o: output score file  
       -v: verbose'
