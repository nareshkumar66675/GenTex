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

#### 1. Generate Text
```
Text Generation/Prediction using HMM
1. Generate Text Sequence
2. Predict Text
3. Re-Train Model
Any other key to exit
Select one option from above : 1
```
#### 2. Generate Text Output
```
Generated Text from Text Corpus

**************

like the power of , walkd i on remote acquaintance were of the illsheathed knife , hoofs were to no cross feet , , friends , to her shall , her these blood , butchery of in . his her lately of of english shall king henry , as which , , we with our a hundred , peace close , , his master fields of paces those . knife and the shock of and henry , , meet , the palace . john all . in of shall womb in wellbeseeming ranks , no more fight , like the nature against so advantage whose childrens those in . , to of , . i of as we and accents walter english shall now afar bred , mothers , pagans entrance soil blessed for the war heaven whose those blunt with , and are ranks moulded , , like the palace


**************
```
#### 3. Text Prediction
```
1. Generate Text Sequence
2. Predict Text
3. Re-Train Model
Any other key to exit
Select one option from above : 2
```


#### 4. Text Prediction Output
```
Text Prediction
Enter a sequence of words: Those Shock

The best prediction is 'those blessed and'
```

#### 5. Model Retrain
```
1. Generate Text Sequence
2. Predict Text
3. Re-Train Model
Any other key to exit
Select one option from above : 3
Retraining Model
Model Retraining Completed
```


# Project Struture

##### **GenTex.py** 
- Main Startup File.
##### **Helper**
- **HMMParam** - A Class object which holds HMM Parameters
- **TextHelper** - A Helper class to do operations on text and hold the tokens
- **Model** - A Class which holds HMMParam, Text and created Probability Values. Used to store the data as pickle
- **ViterbiHelper** - A Helper, which contains all functions related to algorithm
##### Models
- **ModelData** - Simple Model Data File
- **ModelData1** - Large Model Data File. Dont use unless its very necessary.
##### DataSet
-- alllines.txt
-- lorem.txt - Cut down version of alllines.txt



  
