[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/PRX56bbL)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=20626961&assignment_repo_type=AssignmentRepo)
# ExploratoryDataAnalysisPart1
Homework 1 for US-COSC-3337

# COSC 3337 - Data Science I 
## Exploratory Data Analysis Assignment part 1

### Due Date: September 21 (Sunday), 11:59 PM 

#### The goal of this assignment is to:
1. Learn how to pre-process datasets using statistics and visualizations
2. Learn how to interpret computed statistics and visualization to gain an understanding of the dataset 
3. Learn properties of the dataset and its characteristics that may lend to understanding of expected results from data analysis 
####

#### Dataset - winequality
**Intended Outcome:**
The objective of Assignment 1 is to explore the relationships between different 
wine attributes and their potential impact on quality. You will gain practical 
experience in data visualization, exploration, and statistical analysis.

#### Dataset: winequality Data Set
The dataset characterizes specific variety of red wine and provides 
a range of physiochemical and sensory measurements along with indicators of 
preceived quality.

**Attributes:**

- **fixed acidity**: based on tartaric acid (g / dm^3)
- **volatile acidity**: based on acetic acid (g / dm^3)
- **citric acid**: based on citric acid (g / dm^3)
- **residual sugar**: based on residual sugar (g / dm^3)
- **chlorides**: based on chlorides (g / dm^3)
- **free sulphur dioxide**: based on SO2 (mg / dm^3)
- **total sulphur dioxide**: based on SO2 (mg / dm^3)
- **density**: based on density (g / cm^3)
- **pH**: based on pH
- **sulphates**: based on potassium sulphate (g / dm^3)
- **alchohol**: based on alcohol (% vol.)
- **quality**: based on sensory data (score between 0 and 10)

This dataset is a (11+1)D dataset.  The quality attribute can be used
to generate a categorical label based on fixed values.  As an example,
- *Class labels for 'wine quality':*
    - Bad: quality < 4
    - Good: 4 < quality <= 7
    - Very Good: quality > 7

#### Assignment Tasks ####

1. Compute summary statistics of each attribute. What are your two favorite statistical measures, explain why you like them and why are they important for this dataset? 
2. Compute the correlations for each of the pair of attributes
available in the dataset. Interpret the statistical findings.
3. Create a scatter plot for the attributes *residual sugar* and *pH* and interpret the plot.
4. Create a scatter plot for the attributes *fixed acidity* and *citric acid* and interpret the plot.
5. Create histogram of the *quality* attribute and interpret the resulting histogram.
6. Create box plots for *alcohol* and *pH* attributes for the instances of the 
3 *quality* class (Bad, Good, and Very Good), and a box plot for all instances in the dataset.  
Interpret the resulting box plots. Compare the two box plots
7. Write a brief conclusion summarizing the most important findings of this task; in particular, address the findings obtained related to predicting the quality of red wine. 

**Note:**

You are provided a shell python file with the tasks above in which you are to
develop your code.  Basic required python packages you will need are also specified
at start of the file via import statements.  

**Submission Instructions:**

Please push and commit your developed solution to the github repository.  Please ensure
that your Github username, name, and PSID are filled in at the start of the file.
Please ensure that all required plots are generated while running your solution and that
your interpretations and conclusions, as required for each task, are included as comments
after each task listed in the shell python file.

-----------------------

<sub><sup>
License: Property of Quantitative Imaging Laboratory (QIL), Department of Computer Science, University of Houston. This software is property of the QIL, and should not be distributed, reproduced, or shared online, without the permission of the author This software is intended to be used by students of the Data Programming course offered at University of Houston. The contents are not to be reproduced and shared with anyone with out the permission of the author. The contents are not to be posted on any online public hosting websites without the permission of the author. The software is cloned and is available to the students for the duration of the course. At the end of the semester, the Github organization is reset and hence all the existing repositories are reset/deleted, to accommodate the next batch of students.
</sub></sup>

