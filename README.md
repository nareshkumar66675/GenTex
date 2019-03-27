# GenTex

It is an ML Project using **Hidden Markov Model**.


# Overview

  - Given a Text Corpus, it tokenizes and create a HMM Model.
  - It uses Forward and Backward Alogorithm to find the probabilities, which is then used for Text Generation.
  - Using the Created Model, Text prediction is implemented using Viterbi Algorithm.

# Dataset Used
- Shakespeare Plays : https://www.kaggle.com/kingburrito666/shakespeare-plays
    - AllLines.txt - This file contains the text corpus for model building


# Approach

### Approach 1
#### Text Preprocess
- Remove all punctuations except comma and period.
- Tokenize and Label Encode all words

#### HMM Parameters
##### 1) Initial Probability
- It contains the probability of each words given in the text corpus.
##### 2) Transition Probability Matrix
- This matrix accounts for the transition between the states.
- Each **Unique Word** is considered as state. Basically the Vocabulary.
- A Matrix is created which holds the probability of state transition from *Word A to Word B*
##### 3) Emission Probability Matrix
- This matrix accounts for the Emission between the observations and states.
- Here the observations are the next words. *i.e How are you* --> **are** is the observation and **how** is the state. 
- A Matrix is created which holds the probability of emission of each observations to the states.
 
##### Laplace Smoothing 
- Since not every word has a state transition there will be a lot of 0 values.
- To account for that, Lapalace Smoothing is used with Lamba = 1.
##### Text Generation 
- Based on the HMM Parameters, text is generated.
- Forward and Backward Algorithm is used to find the probabilities of each states.
- Based on these probabilities, we select the word based on maximum probability. 
- These words are then used to form the Text.

##### Text Prediction
- Based on the HMM Parameters, text is predicted.
- Viterbi alogorithm is feeded with the HMM Parameters and the new observations enetered by user.
- Best sequence of words is returned by the algorithm.

#### Approach 2
- This is a test approach which is similar to the Approach 1.
- Here the hidden states are characters **a-z**.
- We categorize each word in the text corpus to any one of the alphabets based on the starting letter of the word.
- Unfortunately, results weren't promising. So, did not proceed with this approach.

Note:
    Due to the complexity of the algorithm, entire text corpus is not used.

# Architecture

![FlowChart](https://raw.githubusercontent.com/nareshkumar66675/GenTex/master/Others/GenTex.png "FlowChart") 



# Installation
```
1. Clone the Repository or Download the Project
2. Navigate to the rool folder
3. Execute 'python GenTex.py'
```



# Sample Execution

#### 1. Select DataSet
```
Expectation Maximization using Gaussian Mixture Model
1.Liver
2.Wine
3.Custom(File Path Needed)
Select one Dataset from above : 1
```
#### 2. Select Column
```
Available Columns
0  Age
1  Total_Bilirubin
2  Direct_Bilirubin
3  Alkaline_Phosphotase
4  Alamine_Aminotransferase
5  Aspartate_Aminotransferase
6  Total_Protiens
7  Albumin
8  Albumin_and_Globulin_Ratio
9  Dataset
Select one Column(Enter number): 7
```
#### 3. Show Histogram and Find Optimal Cluster
```
Close the Graph to continue.
Checking feasibility of Cluster:2
Checking feasibility of Cluster:3
Optimal Cluster is 3
```
![AlbuminHistogram](https://raw.githubusercontent.com/nareshkumar66675/ExpMaxML/master/Others/AlbuminHist.png "AlbuminHistogram") 

#### 4. EM Iteration using GMM
```
*******  Iteration 0   *********
Printing Mean
2.355348573822168, 3.1193685474989397, 3.9317409062682755
Log Likelyhood :-151537.16537436945
*******  Iteration 1   *********
Printing Mean
2.356583558295428, 3.117896594822594, 3.928358017839534
Log Likelyhood :-151426.94519112387
0.0
*******  Iteration 2   *********
Printing Mean
2.356504406993411, 3.117180941618275, 3.9271702188672144
Log Likelyhood :-151399.42224758756
0.7502912558674304
..
...
.....
```

#### 5. Final Result and Likelihood Graph
```
Final Mean Values for each cluster:
Cluster 0  : 2.356364851826904
Cluster 1  : 3.116288502073406
Cluster 2  : 3.925730588575203
```
![Likelihood](https://raw.githubusercontent.com/nareshkumar66675/ExpMaxML/master/Others/Convergence.png "Likelihood") 

# Project Struture
##### ExpMaxML
- **ExpMaxML.py** - Main Startup File.
- **/StatFunctions**
    - EM - Methods related to Expectation Maximization
    - PDF - Distribution Implemenataion
    - StatObj - Custom Stat Class
##### Notebooks
- **Liver Analysis** - Very Basic Liver Analysis
##### DataSet
-- indian_liver_patient.csv
-- winequality-red.csv


  
