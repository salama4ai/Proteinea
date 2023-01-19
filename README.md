Deep Learning challenge. Sent to Appicants for Deep Learning Engineer Full Time position at Proteinea firm.
## Problem statement
For this problem, i were asked to generate a **date** given a set of conditions, using any neural network architecture i would like. my input (x) is the conditions on the date, and the output (y) is ANY date that complies with those conditions. This means that, like any generative model, there are many right answers per input x.
[here](https://github.com/salama4ai/Proteinea/blob/main/problem%20statement/Deep%20Learning%20Challenge.pdf) is the complete problem statement 

-**Tools:** Sklearn , Matplotlib , Numpy , Pandas, pytorch, imblearn, calendar
        
-**steps:** 
+ started by [Exploratory data analysis, then preprocessing](https://github.com/salama4ai/Proteinea/blob/main/model/preprocessing.ipynb)
+ then divide the problem into 2 subproblems:- 
	+ [1- predicting the year submodel](https://github.com/salama4ai/Proteinea/blob/main/model/training_years.ipynb)
	+ [2- predicting the day submodel](https://github.com/salama4ai/Proteinea/blob/main/model/training_days.ipynb)
    
	i started by additional preprocessing step according to the submodel needs, then separete training and test sets  fix the imbalanced data by oversampling(only on training set), then prepare data for neural network, and made traing steps finally save the results to csv file, and the trained model
    
#
this work done by Ahmed Salama at NOV-2022

salama4ai@gmail.com

[github.com/salama4ai](https://www.github.com/salama4ai/)

[linkedin.com/in/salama4ai](https://www.linkedin.com/in/salama4ai/)
