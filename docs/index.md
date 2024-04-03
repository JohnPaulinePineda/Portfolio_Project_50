***
# Supervised Learning : Exploring Regularization Approaches for Controlling Model Complexity Through Weight Penalization for Neural Network Classification 

***
### John Pauline Pineda <br> <br> *April 13, 2024*
***

* [**1. Table of Contents**](#TOC)
    * [1.1 Data Background](#1.1)
    * [1.2 Data Description](#1.2)
    * [1.3 Data Quality Assessment](#1.3)
    * [1.4 Data Preprocessing](#1.4)
        * [1.4.1 Data Cleaning](#1.4.1)
        * [1.4.2 Missing Data Imputation](#1.4.2)
        * [1.4.3 Outlier Treatment](#1.4.3)
        * [1.4.4 Collinearity](#1.4.4)
        * [1.4.5 Shape Transformation](#1.4.5)
        * [1.4.6 Centering and Scaling](#1.4.6)
        * [1.4.7 Data Encoding](#1.4.7)
        * [1.4.8 Preprocessed Data Description](#1.4.8)
    * [1.5 Data Exploration](#1.5)
        * [1.5.1 Exploratory Data Analysis](#1.5.1)
        * [1.5.2 Hypothesis Testing](#1.5.2)
    * [1.6 Neural Network Classification Gradient and Weight Updates](#1.6)
        * [1.6.1 Premodelling Data Description](#1.6.1)
        * [1.6.2 Stochastic Gradient Descent Optimization](#1.6.2)
        * [1.6.3 Adaptive Moment Estimation Optimization](#1.6.3)
        * [1.6.4 Adaptive Gradient Algorithm Optimization](#1.6.4)
        * [1.6.5 AdaDelta Optimization](#1.6.5)
        * [1.6.6 Layer-wise Optimized Non-convex Optimization](#1.6.6)
        * [1.6.7 Root Mean Square Propagation Optimization](#1.6.7)
    * [1.7 Consolidated Findings](#1.7)
* [**2. Summary**](#Summary)   
* [**3. References**](#References)

***

# 1. Table of Contents <a class="anchor" id="TOC"></a>

This project manually implements the **Stochastic Gradient Descent Optimization**, **Adaptive Moment Estimation Optimization**, **Adaptive Gradient Optimization**, **AdaDelta Optimization**, **Layer-wise Optimized Non-convex Optimization** and **Root Mean Square Propagation Optimization** algorithms using various helpful packages in <mark style="background-color: #CCECFF"><b>Python</b></mark> with fixed values applied for the learning rate and iteration count parameters to optimally update the gradients and weights of an artificial neural network classification model. The cost function and classification accuracy optimization profiles of the different optimization algorithms were compared. All results were consolidated in a [<span style="color: #FF0000"><b>Summary</b></span>](#Summary) presented at the end of the document.

[Artificial Neural Network](https://link.springer.com/book/10.1007/978-0-387-84858-7), in the context of categorical response prediction, consists of interconnected nodes called neurons organized in layers. The model architecture involves an input layer which receives the input data, with each neuron representing a feature or attribute of the data; hidden layers which perform computations on the input data through weighted connections between neurons and apply activation functions to produce outputs; and the output layer which produces the final predictions equal to the number of classes, each representing the probability of the input belonging to a particular class, based on the computations performed in the hidden layers. Neurons within adjacent layers are connected by weighted connections. Each connection has an associated weight that determines the strength of influence one neuron has on another. These weights are adjusted during the training process to enable the network to learn from the input data and make accurate predictions. Activation functions introduce non-linearities into the network, allowing it to learn complex relationships between inputs and outputs. The training process involves presenting input data along with corresponding target outputs to the network and adjusting the weights to minimize the difference between the predicted outputs and the actual targets which is typically performed through optimization algorithms such as gradient descent and backpropagation. The training process iteratively updates the weights until the model's predictions closely match the target outputs.

[Backpropagation](https://link.springer.com/book/10.1007/978-0-387-84858-7) and [Weight Update](https://link.springer.com/book/10.1007/978-0-387-84858-7), in the context of an artificial neural network, involve the process of iteratively adjusting the weights of the connections between neurons in the network to minimize the difference between the predicted and the actual target responses. Input data is fed into the neural network, and it propagates through the network layer by layer, starting from the input layer, through hidden layers, and ending at the output layer. At each neuron, the weighted sum of inputs is calculated, followed by the application of an activation function to produce the neuron's output. Once the forward pass is complete, the network's output is compared to the actual target output. The difference between the predicted output and the actual output is quantified using a loss function, which measures the discrepancy between the predicted and actual values. Common loss functions for classification tasks include cross-entropy loss. During the backward pass, the error is propagated backward through the network to compute the gradients of the loss function with respect to each weight in the network. This is achieved using the chain rule of calculus, which allows the error to be decomposed and distributed backward through the network. The gradients quantify how much a change in each weight would affect the overall error of the network. Once the gradients are computed, the weights are updated in the opposite direction of the gradient to minimize the error. This update is typically performed using an optimization algorithm such as gradient descent, which adjusts the weights in proportion to their gradients and a learning rate hyperparameter. The learning rate determines the size of the step taken in the direction opposite to the gradient. These steps are repeated for multiple iterations (epochs) over the training data. As the training progresses, the weights are adjusted iteratively to minimize the error, leading to a neural network model that accurately classifies input data.

[Optimization Algorithms](https://link.springer.com/book/10.1007/978-0-387-84858-7), in the context of neural network classification, are methods used to adjust the parameters (weights and biases) of a neural network during the training process in order to minimize a predefined loss function. The primary goal of these algorithms is to optimize the performance of the neural network by iteratively updating its parameters based on the feedback provided by the training data. Optimization algorithms play a critical role in the training of neural networks because they determine how effectively the network learns from the data and how quickly it converges to an optimal solution. These algorithms are significant during model development in improving model accuracy (optimization algorithms help improve the accuracy of neural network models by minimizing the classification error on the training data), enhancing generalization (by minimizing the loss function during training, optimization algorithms aim to generalize well to unseen data, thereby improving the model's ability to make accurate predictions on new inputs), reducing training time (efficient optimization algorithms can accelerate the convergence of the training process, leading to shorter training times for neural networks), handling complex data (since neural networks often deal with high-dimensional and non-linear data, optimization algorithms enable neural networks to effectively learn complex patterns and relationships within the data, leading to improved classification performance) and adapting to variations in data (optimization algorithms can adapt the model's parameters based on variations in the training data, ensuring robustness and stability in the face of different input distributions or data characteristics).


## 1.1. Data Background <a class="anchor" id="1.1"></a>

Datasets used for the analysis were separately gathered and consolidated from various sources including: 
1. Cancer Rates from [World Population Review](https://worldpopulationreview.com/country-rankings/cancer-rates-by-country)
2. Social Protection and Labor Indicator from [World Bank](https://data.worldbank.org/topic/social-protection-and-labor?view=chart)
3. Education Indicator from [World Bank](https://data.worldbank.org/topic/education?view=chart)
4. Economy and Growth Indicator from [World Bank](https://data.worldbank.org/topic/economy-and-growth?view=chart)
5. Environment Indicator from [World Bank](https://data.worldbank.org/topic/environment?view=chart)
6. Climate Change Indicator from [World Bank](https://data.worldbank.org/topic/climate-change?view=chart)
7. Agricultural and Rural Development Indicator from [World Bank](https://data.worldbank.org/topic/agriculture-and-rural-development?view=chart)
8. Social Development Indicator from [World Bank](https://data.worldbank.org/topic/social-development?view=chart)
9. Health Indicator from [World Bank](https://data.worldbank.org/topic/health?view=chart)
10. Science and Technology Indicator from [World Bank](https://data.worldbank.org/topic/science-and-technology?view=chart)
11. Urban Development Indicator from [World Bank](https://data.worldbank.org/topic/urban-development?view=chart)
12. Human Development Indices from [Human Development Reports](https://hdr.undp.org/data-center/human-development-index#/indicies/HDI)
13. Environmental Performance Indices from [Yale Center for Environmental Law and Policy](https://epi.yale.edu/epi-results/2022/component/epi)

This study hypothesized that various global development indicators and indices influence cancer rates across countries.

The target variable for the study is:
* <span style="color: #FF0000">CANRAT</span> - Dichotomized category based on age-standardized cancer rates, per 100K population (2022)

The predictor variables for the study are:
* <span style="color: #FF0000">GDPPER</span> - GDP per person employed, current US Dollars (2020)
* <span style="color: #FF0000">URBPOP</span> - Urban population, % of total population (2020)
* <span style="color: #FF0000">PATRES</span> - Patent applications by residents, total count (2020)
* <span style="color: #FF0000">RNDGDP</span> - Research and development expenditure, % of GDP (2020)
* <span style="color: #FF0000">POPGRO</span> - Population growth, annual % (2020)
* <span style="color: #FF0000">LIFEXP</span> - Life expectancy at birth, total in years (2020)
* <span style="color: #FF0000">TUBINC</span> - Incidence of tuberculosis, per 100K population (2020)
* <span style="color: #FF0000">DTHCMD</span> - Cause of death by communicable diseases and maternal, prenatal and nutrition conditions,  % of total (2019)
* <span style="color: #FF0000">AGRLND</span> - Agricultural land,  % of land area (2020)
* <span style="color: #FF0000">GHGEMI</span> - Total greenhouse gas emissions, kt of CO2 equivalent (2020)
* <span style="color: #FF0000">RELOUT</span> - Renewable electricity output, % of total electricity output (2015)
* <span style="color: #FF0000">METEMI</span> - Methane emissions, kt of CO2 equivalent (2020)
* <span style="color: #FF0000">FORARE</span> - Forest area, % of land area (2020)
* <span style="color: #FF0000">CO2EMI</span> - CO2 emissions, metric tons per capita (2020)
* <span style="color: #FF0000">PM2EXP</span> - PM2.5 air pollution, population exposed to levels exceeding WHO guideline value,  % of total (2017)
* <span style="color: #FF0000">POPDEN</span> - Population density, people per sq. km of land area (2020)
* <span style="color: #FF0000">GDPCAP</span> - GDP per capita, current US Dollars (2020)
* <span style="color: #FF0000">ENRTER</span> - Tertiary school enrollment, % gross (2020)
* <span style="color: #FF0000">HDICAT</span> - Human development index, ordered category (2020)
* <span style="color: #FF0000">EPISCO</span> - Environment performance index , score (2022)


## 1.2. Data Description <a class="anchor" id="1.2"></a>

1. The dataset is comprised of:
    * **177 rows** (observations)
    * **22 columns** (variables)
        * **1/22 metadata** (object)
            * <span style="color: #FF0000">COUNTRY</span>
        * **1/22 target** (categorical)
             * <span style="color: #FF0000">CANRAT</span>
        * **19/22 predictor** (numeric)
             * <span style="color: #FF0000">GDPPER</span>
             * <span style="color: #FF0000">URBPOP</span>
             * <span style="color: #FF0000">PATRES</span>
             * <span style="color: #FF0000">RNDGDP</span>
             * <span style="color: #FF0000">POPGRO</span>
             * <span style="color: #FF0000">LIFEXP</span>
             * <span style="color: #FF0000">TUBINC</span>
             * <span style="color: #FF0000">DTHCMD</span>
             * <span style="color: #FF0000">AGRLND</span>
             * <span style="color: #FF0000">GHGEMI</span>
             * <span style="color: #FF0000">RELOUT</span>
             * <span style="color: #FF0000">METEMI</span>
             * <span style="color: #FF0000">FORARE</span>
             * <span style="color: #FF0000">CO2EMI</span>
             * <span style="color: #FF0000">PM2EXP</span>
             * <span style="color: #FF0000">POPDEN</span>
             * <span style="color: #FF0000">GDPCAP</span>
             * <span style="color: #FF0000">ENRTER</span>
             * <span style="color: #FF0000">EPISCO</span>
        * **1/22 predictor** (categorical)
             * <span style="color: #FF0000">HDICAT</span>


```python
##################################
# Loading Python Libraries
##################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
%matplotlib inline

from operator import add,mul,truediv
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from scipy import stats
```


```python
##################################
# Loading the dataset
##################################
cancer_rate = pd.read_csv('CategoricalCancerRates.csv')
```


```python
##################################
# Performing a general exploration of the dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate.shape)
```

    Dataset Dimensions: 
    


    (177, 22)



```python
##################################
# Listing the column names and data types
##################################
print('Column Names and Data Types:')
display(cancer_rate.dtypes)
```

    Column Names and Data Types:
    


    COUNTRY     object
    CANRAT      object
    GDPPER     float64
    URBPOP     float64
    PATRES     float64
    RNDGDP     float64
    POPGRO     float64
    LIFEXP     float64
    TUBINC     float64
    DTHCMD     float64
    AGRLND     float64
    GHGEMI     float64
    RELOUT     float64
    METEMI     float64
    FORARE     float64
    CO2EMI     float64
    PM2EXP     float64
    POPDEN     float64
    ENRTER     float64
    GDPCAP     float64
    HDICAT      object
    EPISCO     float64
    dtype: object



```python
##################################
# Taking a snapshot of the dataset
##################################
cancer_rate.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>COUNTRY</th>
      <th>CANRAT</th>
      <th>GDPPER</th>
      <th>URBPOP</th>
      <th>PATRES</th>
      <th>RNDGDP</th>
      <th>POPGRO</th>
      <th>LIFEXP</th>
      <th>TUBINC</th>
      <th>DTHCMD</th>
      <th>...</th>
      <th>RELOUT</th>
      <th>METEMI</th>
      <th>FORARE</th>
      <th>CO2EMI</th>
      <th>PM2EXP</th>
      <th>POPDEN</th>
      <th>ENRTER</th>
      <th>GDPCAP</th>
      <th>HDICAT</th>
      <th>EPISCO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Australia</td>
      <td>High</td>
      <td>98380.63601</td>
      <td>86.241</td>
      <td>2368.0</td>
      <td>NaN</td>
      <td>1.235701</td>
      <td>83.200000</td>
      <td>7.2</td>
      <td>4.941054</td>
      <td>...</td>
      <td>13.637841</td>
      <td>131484.763200</td>
      <td>17.421315</td>
      <td>14.772658</td>
      <td>24.893584</td>
      <td>3.335312</td>
      <td>110.139221</td>
      <td>51722.06900</td>
      <td>VH</td>
      <td>60.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>New Zealand</td>
      <td>High</td>
      <td>77541.76438</td>
      <td>86.699</td>
      <td>348.0</td>
      <td>NaN</td>
      <td>2.204789</td>
      <td>82.256098</td>
      <td>7.2</td>
      <td>4.354730</td>
      <td>...</td>
      <td>80.081439</td>
      <td>32241.937000</td>
      <td>37.570126</td>
      <td>6.160799</td>
      <td>NaN</td>
      <td>19.331586</td>
      <td>75.734833</td>
      <td>41760.59478</td>
      <td>VH</td>
      <td>56.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ireland</td>
      <td>High</td>
      <td>198405.87500</td>
      <td>63.653</td>
      <td>75.0</td>
      <td>1.23244</td>
      <td>1.029111</td>
      <td>82.556098</td>
      <td>5.3</td>
      <td>5.684596</td>
      <td>...</td>
      <td>27.965408</td>
      <td>15252.824630</td>
      <td>11.351720</td>
      <td>6.768228</td>
      <td>0.274092</td>
      <td>72.367281</td>
      <td>74.680313</td>
      <td>85420.19086</td>
      <td>VH</td>
      <td>57.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>United States</td>
      <td>High</td>
      <td>130941.63690</td>
      <td>82.664</td>
      <td>269586.0</td>
      <td>3.42287</td>
      <td>0.964348</td>
      <td>76.980488</td>
      <td>2.3</td>
      <td>5.302060</td>
      <td>...</td>
      <td>13.228593</td>
      <td>748241.402900</td>
      <td>33.866926</td>
      <td>13.032828</td>
      <td>3.343170</td>
      <td>36.240985</td>
      <td>87.567657</td>
      <td>63528.63430</td>
      <td>VH</td>
      <td>51.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Denmark</td>
      <td>High</td>
      <td>113300.60110</td>
      <td>88.116</td>
      <td>1261.0</td>
      <td>2.96873</td>
      <td>0.291641</td>
      <td>81.602439</td>
      <td>4.1</td>
      <td>6.826140</td>
      <td>...</td>
      <td>65.505925</td>
      <td>7778.773921</td>
      <td>15.711000</td>
      <td>4.691237</td>
      <td>56.914456</td>
      <td>145.785100</td>
      <td>82.664330</td>
      <td>60915.42440</td>
      <td>VH</td>
      <td>77.9</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
##################################
# Setting the levels of the categorical variables
##################################
cancer_rate['CANRAT'] = cancer_rate['CANRAT'].astype('category')
cancer_rate['CANRAT'] = cancer_rate['CANRAT'].cat.set_categories(['Low', 'High'], ordered=True)
cancer_rate['HDICAT'] = cancer_rate['HDICAT'].astype('category')
cancer_rate['HDICAT'] = cancer_rate['HDICAT'].cat.set_categories(['L', 'M', 'H', 'VH'], ordered=True)
```


```python
##################################
# Performing a general exploration of the numeric variables
##################################
print('Numeric Variable Summary:')
display(cancer_rate.describe(include='number').transpose())
```

    Numeric Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GDPPER</th>
      <td>165.0</td>
      <td>45284.424283</td>
      <td>3.941794e+04</td>
      <td>1718.804896</td>
      <td>13545.254510</td>
      <td>34024.900890</td>
      <td>66778.416050</td>
      <td>2.346469e+05</td>
    </tr>
    <tr>
      <th>URBPOP</th>
      <td>174.0</td>
      <td>59.788121</td>
      <td>2.280640e+01</td>
      <td>13.345000</td>
      <td>42.432750</td>
      <td>61.701500</td>
      <td>79.186500</td>
      <td>1.000000e+02</td>
    </tr>
    <tr>
      <th>PATRES</th>
      <td>108.0</td>
      <td>20607.388889</td>
      <td>1.340683e+05</td>
      <td>1.000000</td>
      <td>35.250000</td>
      <td>244.500000</td>
      <td>1297.750000</td>
      <td>1.344817e+06</td>
    </tr>
    <tr>
      <th>RNDGDP</th>
      <td>74.0</td>
      <td>1.197474</td>
      <td>1.189956e+00</td>
      <td>0.039770</td>
      <td>0.256372</td>
      <td>0.873660</td>
      <td>1.608842</td>
      <td>5.354510e+00</td>
    </tr>
    <tr>
      <th>POPGRO</th>
      <td>174.0</td>
      <td>1.127028</td>
      <td>1.197718e+00</td>
      <td>-2.079337</td>
      <td>0.236900</td>
      <td>1.179959</td>
      <td>2.031154</td>
      <td>3.727101e+00</td>
    </tr>
    <tr>
      <th>LIFEXP</th>
      <td>174.0</td>
      <td>71.746113</td>
      <td>7.606209e+00</td>
      <td>52.777000</td>
      <td>65.907500</td>
      <td>72.464610</td>
      <td>77.523500</td>
      <td>8.456000e+01</td>
    </tr>
    <tr>
      <th>TUBINC</th>
      <td>174.0</td>
      <td>105.005862</td>
      <td>1.367229e+02</td>
      <td>0.770000</td>
      <td>12.000000</td>
      <td>44.500000</td>
      <td>147.750000</td>
      <td>5.920000e+02</td>
    </tr>
    <tr>
      <th>DTHCMD</th>
      <td>170.0</td>
      <td>21.260521</td>
      <td>1.927333e+01</td>
      <td>1.283611</td>
      <td>6.078009</td>
      <td>12.456279</td>
      <td>36.980457</td>
      <td>6.520789e+01</td>
    </tr>
    <tr>
      <th>AGRLND</th>
      <td>174.0</td>
      <td>38.793456</td>
      <td>2.171551e+01</td>
      <td>0.512821</td>
      <td>20.130276</td>
      <td>40.386649</td>
      <td>54.013754</td>
      <td>8.084112e+01</td>
    </tr>
    <tr>
      <th>GHGEMI</th>
      <td>170.0</td>
      <td>259582.709895</td>
      <td>1.118550e+06</td>
      <td>179.725150</td>
      <td>12527.487367</td>
      <td>41009.275980</td>
      <td>116482.578575</td>
      <td>1.294287e+07</td>
    </tr>
    <tr>
      <th>RELOUT</th>
      <td>153.0</td>
      <td>39.760036</td>
      <td>3.191492e+01</td>
      <td>0.000296</td>
      <td>10.582691</td>
      <td>32.381668</td>
      <td>63.011450</td>
      <td>1.000000e+02</td>
    </tr>
    <tr>
      <th>METEMI</th>
      <td>170.0</td>
      <td>47876.133575</td>
      <td>1.346611e+05</td>
      <td>11.596147</td>
      <td>3662.884908</td>
      <td>11118.976025</td>
      <td>32368.909040</td>
      <td>1.186285e+06</td>
    </tr>
    <tr>
      <th>FORARE</th>
      <td>173.0</td>
      <td>32.218177</td>
      <td>2.312001e+01</td>
      <td>0.008078</td>
      <td>11.604388</td>
      <td>31.509048</td>
      <td>49.071780</td>
      <td>9.741212e+01</td>
    </tr>
    <tr>
      <th>CO2EMI</th>
      <td>170.0</td>
      <td>3.751097</td>
      <td>4.606479e+00</td>
      <td>0.032585</td>
      <td>0.631924</td>
      <td>2.298368</td>
      <td>4.823496</td>
      <td>3.172684e+01</td>
    </tr>
    <tr>
      <th>PM2EXP</th>
      <td>167.0</td>
      <td>91.940595</td>
      <td>2.206003e+01</td>
      <td>0.274092</td>
      <td>99.627134</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>1.000000e+02</td>
    </tr>
    <tr>
      <th>POPDEN</th>
      <td>174.0</td>
      <td>200.886765</td>
      <td>6.453834e+02</td>
      <td>2.115134</td>
      <td>27.454539</td>
      <td>77.983133</td>
      <td>153.993650</td>
      <td>7.918951e+03</td>
    </tr>
    <tr>
      <th>ENRTER</th>
      <td>116.0</td>
      <td>49.994997</td>
      <td>2.970619e+01</td>
      <td>2.432581</td>
      <td>22.107195</td>
      <td>53.392460</td>
      <td>71.057467</td>
      <td>1.433107e+02</td>
    </tr>
    <tr>
      <th>GDPCAP</th>
      <td>170.0</td>
      <td>13992.095610</td>
      <td>1.957954e+04</td>
      <td>216.827417</td>
      <td>1870.503029</td>
      <td>5348.192875</td>
      <td>17421.116227</td>
      <td>1.173705e+05</td>
    </tr>
    <tr>
      <th>EPISCO</th>
      <td>165.0</td>
      <td>42.946667</td>
      <td>1.249086e+01</td>
      <td>18.900000</td>
      <td>33.000000</td>
      <td>40.900000</td>
      <td>50.500000</td>
      <td>7.790000e+01</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Performing a general exploration of the object variable
##################################
print('Object Variable Summary:')
display(cancer_rate.describe(include='object').transpose())
```

    Object Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>COUNTRY</th>
      <td>177</td>
      <td>177</td>
      <td>Australia</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Performing a general exploration of the categorical variables
##################################
print('Categorical Variable Summary:')
display(cancer_rate.describe(include='category').transpose())
```

    Categorical Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CANRAT</th>
      <td>177</td>
      <td>2</td>
      <td>Low</td>
      <td>132</td>
    </tr>
    <tr>
      <th>HDICAT</th>
      <td>167</td>
      <td>4</td>
      <td>VH</td>
      <td>59</td>
    </tr>
  </tbody>
</table>
</div>


## 1.3. Data Quality Assessment <a class="anchor" id="1.3"></a>

Data quality findings based on assessment are as follows:
1. No duplicated rows observed.
2. Missing data noted for 20 variables with Null.Count>0 and Fill.Rate<1.0.
    * <span style="color: #FF0000">RNDGDP</span>: Null.Count = 103, Fill.Rate = 0.418
    * <span style="color: #FF0000">PATRES</span>: Null.Count = 69, Fill.Rate = 0.610
    * <span style="color: #FF0000">ENRTER</span>: Null.Count = 61, Fill.Rate = 0.655
    * <span style="color: #FF0000">RELOUT</span>: Null.Count = 24, Fill.Rate = 0.864
    * <span style="color: #FF0000">GDPPER</span>: Null.Count = 12, Fill.Rate = 0.932
    * <span style="color: #FF0000">EPISCO</span>: Null.Count = 12, Fill.Rate = 0.932
    * <span style="color: #FF0000">HDICAT</span>: Null.Count = 10, Fill.Rate = 0.943
    * <span style="color: #FF0000">PM2EXP</span>: Null.Count = 10, Fill.Rate = 0.943
    * <span style="color: #FF0000">DTHCMD</span>: Null.Count = 7, Fill.Rate = 0.960
    * <span style="color: #FF0000">METEMI</span>: Null.Count = 7, Fill.Rate = 0.960
    * <span style="color: #FF0000">CO2EMI</span>: Null.Count = 7, Fill.Rate = 0.960
    * <span style="color: #FF0000">GDPCAP</span>: Null.Count = 7, Fill.Rate = 0.960
    * <span style="color: #FF0000">GHGEMI</span>: Null.Count = 7, Fill.Rate = 0.960
    * <span style="color: #FF0000">FORARE</span>: Null.Count = 4, Fill.Rate = 0.977
    * <span style="color: #FF0000">TUBINC</span>: Null.Count = 3, Fill.Rate = 0.983
    * <span style="color: #FF0000">AGRLND</span>: Null.Count = 3, Fill.Rate = 0.983
    * <span style="color: #FF0000">POPGRO</span>: Null.Count = 3, Fill.Rate = 0.983
    * <span style="color: #FF0000">POPDEN</span>: Null.Count = 3, Fill.Rate = 0.983
    * <span style="color: #FF0000">URBPOP</span>: Null.Count = 3, Fill.Rate = 0.983
    * <span style="color: #FF0000">LIFEXP</span>: Null.Count = 3, Fill.Rate = 0.983
3. 120 observations noted with at least 1 missing data. From this number, 14 observations reported high Missing.Rate>0.2.
    * <span style="color: #FF0000">COUNTRY=Guadeloupe</span>: Missing.Rate= 0.909
    * <span style="color: #FF0000">COUNTRY=Martinique</span>: Missing.Rate= 0.909
    * <span style="color: #FF0000">COUNTRY=French Guiana</span>: Missing.Rate= 0.909
    * <span style="color: #FF0000">COUNTRY=New Caledonia</span>: Missing.Rate= 0.500
    * <span style="color: #FF0000">COUNTRY=French Polynesia</span>: Missing.Rate= 0.500
    * <span style="color: #FF0000">COUNTRY=Guam</span>: Missing.Rate= 0.500
    * <span style="color: #FF0000">COUNTRY=Puerto Rico</span>: Missing.Rate= 0.409
    * <span style="color: #FF0000">COUNTRY=North Korea</span>: Missing.Rate= 0.227
    * <span style="color: #FF0000">COUNTRY=Somalia</span>: Missing.Rate= 0.227
    * <span style="color: #FF0000">COUNTRY=South Sudan</span>: Missing.Rate= 0.227
    * <span style="color: #FF0000">COUNTRY=Venezuela</span>: Missing.Rate= 0.227
    * <span style="color: #FF0000">COUNTRY=Libya</span>: Missing.Rate= 0.227
    * <span style="color: #FF0000">COUNTRY=Eritrea</span>: Missing.Rate= 0.227
    * <span style="color: #FF0000">COUNTRY=Yemen</span>: Missing.Rate= 0.227
4. Low variance observed for 1 variable with First.Second.Mode.Ratio>5.
    * <span style="color: #FF0000">PM2EXP</span>: First.Second.Mode.Ratio = 53.000
5. No low variance observed for any variable with Unique.Count.Ratio>10.
6. High skewness observed for 5 variables with Skewness>3 or Skewness<(-3).
    * <span style="color: #FF0000">POPDEN</span>: Skewness = +10.267
    * <span style="color: #FF0000">GHGEMI</span>: Skewness = +9.496
    * <span style="color: #FF0000">PATRES</span>: Skewness = +9.284
    * <span style="color: #FF0000">METEMI</span>: Skewness = +5.801
    * <span style="color: #FF0000">PM2EXP</span>: Skewness = -3.141


```python
##################################
# Counting the number of duplicated rows
##################################
cancer_rate.duplicated().sum()
```




    0




```python
##################################
# Gathering the data types for each column
##################################
data_type_list = list(cancer_rate.dtypes)
```


```python
##################################
# Gathering the variable names for each column
##################################
variable_name_list = list(cancer_rate.columns)
```


```python
##################################
# Gathering the number of observations for each column
##################################
row_count_list = list([len(cancer_rate)] * len(cancer_rate.columns))
```


```python
##################################
# Gathering the number of missing data for each column
##################################
null_count_list = list(cancer_rate.isna().sum(axis=0))
```


```python
##################################
# Gathering the number of non-missing data for each column
##################################
non_null_count_list = list(cancer_rate.count())
```


```python
##################################
# Gathering the missing data percentage for each column
##################################
fill_rate_list = map(truediv, non_null_count_list, row_count_list)
```


```python
##################################
# Formulating the summary
# for all columns
##################################
all_column_quality_summary = pd.DataFrame(zip(variable_name_list,
                                              data_type_list,
                                              row_count_list,
                                              non_null_count_list,
                                              null_count_list,
                                              fill_rate_list), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count',                                                 
                                                 'Fill.Rate'])
display(all_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
      <th>Fill.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>COUNTRY</td>
      <td>object</td>
      <td>177</td>
      <td>177</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CANRAT</td>
      <td>category</td>
      <td>177</td>
      <td>177</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GDPPER</td>
      <td>float64</td>
      <td>177</td>
      <td>165</td>
      <td>12</td>
      <td>0.932203</td>
    </tr>
    <tr>
      <th>3</th>
      <td>URBPOP</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PATRES</td>
      <td>float64</td>
      <td>177</td>
      <td>108</td>
      <td>69</td>
      <td>0.610169</td>
    </tr>
    <tr>
      <th>5</th>
      <td>RNDGDP</td>
      <td>float64</td>
      <td>177</td>
      <td>74</td>
      <td>103</td>
      <td>0.418079</td>
    </tr>
    <tr>
      <th>6</th>
      <td>POPGRO</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LIFEXP</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TUBINC</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>9</th>
      <td>DTHCMD</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>10</th>
      <td>AGRLND</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>11</th>
      <td>GHGEMI</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>12</th>
      <td>RELOUT</td>
      <td>float64</td>
      <td>177</td>
      <td>153</td>
      <td>24</td>
      <td>0.864407</td>
    </tr>
    <tr>
      <th>13</th>
      <td>METEMI</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>14</th>
      <td>FORARE</td>
      <td>float64</td>
      <td>177</td>
      <td>173</td>
      <td>4</td>
      <td>0.977401</td>
    </tr>
    <tr>
      <th>15</th>
      <td>CO2EMI</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PM2EXP</td>
      <td>float64</td>
      <td>177</td>
      <td>167</td>
      <td>10</td>
      <td>0.943503</td>
    </tr>
    <tr>
      <th>17</th>
      <td>POPDEN</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ENRTER</td>
      <td>float64</td>
      <td>177</td>
      <td>116</td>
      <td>61</td>
      <td>0.655367</td>
    </tr>
    <tr>
      <th>19</th>
      <td>GDPCAP</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>20</th>
      <td>HDICAT</td>
      <td>category</td>
      <td>177</td>
      <td>167</td>
      <td>10</td>
      <td>0.943503</td>
    </tr>
    <tr>
      <th>21</th>
      <td>EPISCO</td>
      <td>float64</td>
      <td>177</td>
      <td>165</td>
      <td>12</td>
      <td>0.932203</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of columns
# with Fill.Rate < 1.00
##################################
len(all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<1)])
```




    20




```python
##################################
# Identifying the columns
# with Fill.Rate < 1.00
##################################
display(all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<1)].sort_values(by=['Fill.Rate'], ascending=True))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
      <th>Fill.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>RNDGDP</td>
      <td>float64</td>
      <td>177</td>
      <td>74</td>
      <td>103</td>
      <td>0.418079</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PATRES</td>
      <td>float64</td>
      <td>177</td>
      <td>108</td>
      <td>69</td>
      <td>0.610169</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ENRTER</td>
      <td>float64</td>
      <td>177</td>
      <td>116</td>
      <td>61</td>
      <td>0.655367</td>
    </tr>
    <tr>
      <th>12</th>
      <td>RELOUT</td>
      <td>float64</td>
      <td>177</td>
      <td>153</td>
      <td>24</td>
      <td>0.864407</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GDPPER</td>
      <td>float64</td>
      <td>177</td>
      <td>165</td>
      <td>12</td>
      <td>0.932203</td>
    </tr>
    <tr>
      <th>21</th>
      <td>EPISCO</td>
      <td>float64</td>
      <td>177</td>
      <td>165</td>
      <td>12</td>
      <td>0.932203</td>
    </tr>
    <tr>
      <th>20</th>
      <td>HDICAT</td>
      <td>category</td>
      <td>177</td>
      <td>167</td>
      <td>10</td>
      <td>0.943503</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PM2EXP</td>
      <td>float64</td>
      <td>177</td>
      <td>167</td>
      <td>10</td>
      <td>0.943503</td>
    </tr>
    <tr>
      <th>9</th>
      <td>DTHCMD</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>13</th>
      <td>METEMI</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>15</th>
      <td>CO2EMI</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>19</th>
      <td>GDPCAP</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>11</th>
      <td>GHGEMI</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>14</th>
      <td>FORARE</td>
      <td>float64</td>
      <td>177</td>
      <td>173</td>
      <td>4</td>
      <td>0.977401</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TUBINC</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>10</th>
      <td>AGRLND</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>6</th>
      <td>POPGRO</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>17</th>
      <td>POPDEN</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>3</th>
      <td>URBPOP</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LIFEXP</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Identifying the rows
# with Fill.Rate < 0.90
##################################
column_low_fill_rate = all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<0.90)]
```


```python
##################################
# Gathering the metadata labels for each observation
##################################
row_metadata_list = cancer_rate["COUNTRY"].values.tolist()
```


```python
##################################
# Gathering the number of columns for each observation
##################################
column_count_list = list([len(cancer_rate.columns)] * len(cancer_rate))
```


```python
##################################
# Gathering the number of missing data for each row
##################################
null_row_list = list(cancer_rate.isna().sum(axis=1))
```


```python
##################################
# Gathering the missing data percentage for each column
##################################
missing_rate_list = map(truediv, null_row_list, column_count_list)
```


```python
##################################
# Identifying the rows
# with missing data
##################################
all_row_quality_summary = pd.DataFrame(zip(row_metadata_list,
                                           column_count_list,
                                           null_row_list,
                                           missing_rate_list), 
                                        columns=['Row.Name',
                                                 'Column.Count',
                                                 'Null.Count',                                                 
                                                 'Missing.Rate'])
display(all_row_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Row.Name</th>
      <th>Column.Count</th>
      <th>Null.Count</th>
      <th>Missing.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Australia</td>
      <td>22</td>
      <td>1</td>
      <td>0.045455</td>
    </tr>
    <tr>
      <th>1</th>
      <td>New Zealand</td>
      <td>22</td>
      <td>2</td>
      <td>0.090909</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ireland</td>
      <td>22</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>United States</td>
      <td>22</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Denmark</td>
      <td>22</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>172</th>
      <td>Congo Republic</td>
      <td>22</td>
      <td>3</td>
      <td>0.136364</td>
    </tr>
    <tr>
      <th>173</th>
      <td>Bhutan</td>
      <td>22</td>
      <td>2</td>
      <td>0.090909</td>
    </tr>
    <tr>
      <th>174</th>
      <td>Nepal</td>
      <td>22</td>
      <td>2</td>
      <td>0.090909</td>
    </tr>
    <tr>
      <th>175</th>
      <td>Gambia</td>
      <td>22</td>
      <td>4</td>
      <td>0.181818</td>
    </tr>
    <tr>
      <th>176</th>
      <td>Niger</td>
      <td>22</td>
      <td>2</td>
      <td>0.090909</td>
    </tr>
  </tbody>
</table>
<p>177 rows × 4 columns</p>
</div>



```python
##################################
# Counting the number of rows
# with Missing.Rate > 0.00
##################################
len(all_row_quality_summary[(all_row_quality_summary['Missing.Rate']>0.00)])
```




    120




```python
##################################
# Counting the number of rows
# with Missing.Rate > 0.20
##################################
len(all_row_quality_summary[(all_row_quality_summary['Missing.Rate']>0.20)])
```




    14




```python
##################################
# Identifying the rows
# with Missing.Rate > 0.20
##################################
row_high_missing_rate = all_row_quality_summary[(all_row_quality_summary['Missing.Rate']>0.20)]
```


```python
##################################
# Identifying the rows
# with Missing.Rate > 0.20
##################################
display(all_row_quality_summary[(all_row_quality_summary['Missing.Rate']>0.20)].sort_values(by=['Missing.Rate'], ascending=False))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Row.Name</th>
      <th>Column.Count</th>
      <th>Null.Count</th>
      <th>Missing.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35</th>
      <td>Guadeloupe</td>
      <td>22</td>
      <td>20</td>
      <td>0.909091</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Martinique</td>
      <td>22</td>
      <td>20</td>
      <td>0.909091</td>
    </tr>
    <tr>
      <th>56</th>
      <td>French Guiana</td>
      <td>22</td>
      <td>20</td>
      <td>0.909091</td>
    </tr>
    <tr>
      <th>13</th>
      <td>New Caledonia</td>
      <td>22</td>
      <td>11</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>44</th>
      <td>French Polynesia</td>
      <td>22</td>
      <td>11</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>75</th>
      <td>Guam</td>
      <td>22</td>
      <td>11</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Puerto Rico</td>
      <td>22</td>
      <td>9</td>
      <td>0.409091</td>
    </tr>
    <tr>
      <th>85</th>
      <td>North Korea</td>
      <td>22</td>
      <td>6</td>
      <td>0.272727</td>
    </tr>
    <tr>
      <th>132</th>
      <td>Somalia</td>
      <td>22</td>
      <td>6</td>
      <td>0.272727</td>
    </tr>
    <tr>
      <th>168</th>
      <td>South Sudan</td>
      <td>22</td>
      <td>6</td>
      <td>0.272727</td>
    </tr>
    <tr>
      <th>73</th>
      <td>Venezuela</td>
      <td>22</td>
      <td>5</td>
      <td>0.227273</td>
    </tr>
    <tr>
      <th>117</th>
      <td>Libya</td>
      <td>22</td>
      <td>5</td>
      <td>0.227273</td>
    </tr>
    <tr>
      <th>161</th>
      <td>Eritrea</td>
      <td>22</td>
      <td>5</td>
      <td>0.227273</td>
    </tr>
    <tr>
      <th>164</th>
      <td>Yemen</td>
      <td>22</td>
      <td>5</td>
      <td>0.227273</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the dataset
# with numeric columns only
##################################
cancer_rate_numeric = cancer_rate.select_dtypes(include='number')
```


```python
##################################
# Gathering the variable names for each numeric column
##################################
numeric_variable_name_list = cancer_rate_numeric.columns
```


```python
##################################
# Gathering the minimum value for each numeric column
##################################
numeric_minimum_list = cancer_rate_numeric.min()
```


```python
##################################
# Gathering the mean value for each numeric column
##################################
numeric_mean_list = cancer_rate_numeric.mean()
```


```python
##################################
# Gathering the median value for each numeric column
##################################
numeric_median_list = cancer_rate_numeric.median()
```


```python
##################################
# Gathering the maximum value for each numeric column
##################################
numeric_maximum_list = cancer_rate_numeric.max()
```


```python
##################################
# Gathering the first mode values for each numeric column
##################################
numeric_first_mode_list = [cancer_rate[x].value_counts(dropna=True).index.tolist()[0] for x in cancer_rate_numeric]
```


```python
##################################
# Gathering the second mode values for each numeric column
##################################
numeric_second_mode_list = [cancer_rate[x].value_counts(dropna=True).index.tolist()[1] for x in cancer_rate_numeric]
```


```python
##################################
# Gathering the count of first mode values for each numeric column
##################################
numeric_first_mode_count_list = [cancer_rate_numeric[x].isin([cancer_rate[x].value_counts(dropna=True).index.tolist()[0]]).sum() for x in cancer_rate_numeric]
```


```python
##################################
# Gathering the count of second mode values for each numeric column
##################################
numeric_second_mode_count_list = [cancer_rate_numeric[x].isin([cancer_rate[x].value_counts(dropna=True).index.tolist()[1]]).sum() for x in cancer_rate_numeric]
```


```python
##################################
# Gathering the first mode to second mode ratio for each numeric column
##################################
numeric_first_second_mode_ratio_list = map(truediv, numeric_first_mode_count_list, numeric_second_mode_count_list)
```


```python
##################################
# Gathering the count of unique values for each numeric column
##################################
numeric_unique_count_list = cancer_rate_numeric.nunique(dropna=True)
```


```python
##################################
# Gathering the number of observations for each numeric column
##################################
numeric_row_count_list = list([len(cancer_rate_numeric)] * len(cancer_rate_numeric.columns))
```


```python
##################################
# Gathering the unique to count ratio for each numeric column
##################################
numeric_unique_count_ratio_list = map(truediv, numeric_unique_count_list, numeric_row_count_list)
```


```python
##################################
# Gathering the skewness value for each numeric column
##################################
numeric_skewness_list = cancer_rate_numeric.skew()
```


```python
##################################
# Gathering the kurtosis value for each numeric column
##################################
numeric_kurtosis_list = cancer_rate_numeric.kurtosis()
```


```python
numeric_column_quality_summary = pd.DataFrame(zip(numeric_variable_name_list,
                                                numeric_minimum_list,
                                                numeric_mean_list,
                                                numeric_median_list,
                                                numeric_maximum_list,
                                                numeric_first_mode_list,
                                                numeric_second_mode_list,
                                                numeric_first_mode_count_list,
                                                numeric_second_mode_count_list,
                                                numeric_first_second_mode_ratio_list,
                                                numeric_unique_count_list,
                                                numeric_row_count_list,
                                                numeric_unique_count_ratio_list,
                                                numeric_skewness_list,
                                                numeric_kurtosis_list), 
                                        columns=['Numeric.Column.Name',
                                                 'Minimum',
                                                 'Mean',
                                                 'Median',
                                                 'Maximum',
                                                 'First.Mode',
                                                 'Second.Mode',
                                                 'First.Mode.Count',
                                                 'Second.Mode.Count',
                                                 'First.Second.Mode.Ratio',
                                                 'Unique.Count',
                                                 'Row.Count',
                                                 'Unique.Count.Ratio',
                                                 'Skewness',
                                                 'Kurtosis'])
display(numeric_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Minimum</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Maximum</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
      <th>Skewness</th>
      <th>Kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GDPPER</td>
      <td>1718.804896</td>
      <td>45284.424283</td>
      <td>34024.900890</td>
      <td>2.346469e+05</td>
      <td>98380.636010</td>
      <td>42154.178100</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>165</td>
      <td>177</td>
      <td>0.932203</td>
      <td>1.517574</td>
      <td>3.471992</td>
    </tr>
    <tr>
      <th>1</th>
      <td>URBPOP</td>
      <td>13.345000</td>
      <td>59.788121</td>
      <td>61.701500</td>
      <td>1.000000e+02</td>
      <td>100.000000</td>
      <td>52.516000</td>
      <td>2</td>
      <td>1</td>
      <td>2.000000</td>
      <td>173</td>
      <td>177</td>
      <td>0.977401</td>
      <td>-0.210702</td>
      <td>-0.962847</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PATRES</td>
      <td>1.000000</td>
      <td>20607.388889</td>
      <td>244.500000</td>
      <td>1.344817e+06</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>4</td>
      <td>3</td>
      <td>1.333333</td>
      <td>97</td>
      <td>177</td>
      <td>0.548023</td>
      <td>9.284436</td>
      <td>91.187178</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RNDGDP</td>
      <td>0.039770</td>
      <td>1.197474</td>
      <td>0.873660</td>
      <td>5.354510e+00</td>
      <td>1.232440</td>
      <td>0.962180</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>74</td>
      <td>177</td>
      <td>0.418079</td>
      <td>1.396742</td>
      <td>1.695957</td>
    </tr>
    <tr>
      <th>4</th>
      <td>POPGRO</td>
      <td>-2.079337</td>
      <td>1.127028</td>
      <td>1.179959</td>
      <td>3.727101e+00</td>
      <td>1.235701</td>
      <td>1.483129</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>174</td>
      <td>177</td>
      <td>0.983051</td>
      <td>-0.195161</td>
      <td>-0.423580</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LIFEXP</td>
      <td>52.777000</td>
      <td>71.746113</td>
      <td>72.464610</td>
      <td>8.456000e+01</td>
      <td>83.200000</td>
      <td>68.687000</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>174</td>
      <td>177</td>
      <td>0.983051</td>
      <td>-0.357965</td>
      <td>-0.649601</td>
    </tr>
    <tr>
      <th>6</th>
      <td>TUBINC</td>
      <td>0.770000</td>
      <td>105.005862</td>
      <td>44.500000</td>
      <td>5.920000e+02</td>
      <td>12.000000</td>
      <td>7.200000</td>
      <td>4</td>
      <td>3</td>
      <td>1.333333</td>
      <td>131</td>
      <td>177</td>
      <td>0.740113</td>
      <td>1.746333</td>
      <td>2.429368</td>
    </tr>
    <tr>
      <th>7</th>
      <td>DTHCMD</td>
      <td>1.283611</td>
      <td>21.260521</td>
      <td>12.456279</td>
      <td>6.520789e+01</td>
      <td>4.941054</td>
      <td>42.079403</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>170</td>
      <td>177</td>
      <td>0.960452</td>
      <td>0.900509</td>
      <td>-0.691541</td>
    </tr>
    <tr>
      <th>8</th>
      <td>AGRLND</td>
      <td>0.512821</td>
      <td>38.793456</td>
      <td>40.386649</td>
      <td>8.084112e+01</td>
      <td>46.252480</td>
      <td>72.006469</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>174</td>
      <td>177</td>
      <td>0.983051</td>
      <td>0.074000</td>
      <td>-0.926249</td>
    </tr>
    <tr>
      <th>9</th>
      <td>GHGEMI</td>
      <td>179.725150</td>
      <td>259582.709895</td>
      <td>41009.275980</td>
      <td>1.294287e+07</td>
      <td>571903.119900</td>
      <td>3000.932259</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>170</td>
      <td>177</td>
      <td>0.960452</td>
      <td>9.496120</td>
      <td>101.637308</td>
    </tr>
    <tr>
      <th>10</th>
      <td>RELOUT</td>
      <td>0.000296</td>
      <td>39.760036</td>
      <td>32.381668</td>
      <td>1.000000e+02</td>
      <td>100.000000</td>
      <td>13.637841</td>
      <td>3</td>
      <td>1</td>
      <td>3.000000</td>
      <td>151</td>
      <td>177</td>
      <td>0.853107</td>
      <td>0.501088</td>
      <td>-0.981774</td>
    </tr>
    <tr>
      <th>11</th>
      <td>METEMI</td>
      <td>11.596147</td>
      <td>47876.133575</td>
      <td>11118.976025</td>
      <td>1.186285e+06</td>
      <td>131484.763200</td>
      <td>1326.034028</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>170</td>
      <td>177</td>
      <td>0.960452</td>
      <td>5.801014</td>
      <td>38.661386</td>
    </tr>
    <tr>
      <th>12</th>
      <td>FORARE</td>
      <td>0.008078</td>
      <td>32.218177</td>
      <td>31.509048</td>
      <td>9.741212e+01</td>
      <td>17.421315</td>
      <td>8.782159</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>173</td>
      <td>177</td>
      <td>0.977401</td>
      <td>0.519277</td>
      <td>-0.322589</td>
    </tr>
    <tr>
      <th>13</th>
      <td>CO2EMI</td>
      <td>0.032585</td>
      <td>3.751097</td>
      <td>2.298368</td>
      <td>3.172684e+01</td>
      <td>14.772658</td>
      <td>0.972088</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>170</td>
      <td>177</td>
      <td>0.960452</td>
      <td>2.721552</td>
      <td>10.311574</td>
    </tr>
    <tr>
      <th>14</th>
      <td>PM2EXP</td>
      <td>0.274092</td>
      <td>91.940595</td>
      <td>100.000000</td>
      <td>1.000000e+02</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>106</td>
      <td>2</td>
      <td>53.000000</td>
      <td>61</td>
      <td>177</td>
      <td>0.344633</td>
      <td>-3.141557</td>
      <td>9.032386</td>
    </tr>
    <tr>
      <th>15</th>
      <td>POPDEN</td>
      <td>2.115134</td>
      <td>200.886765</td>
      <td>77.983133</td>
      <td>7.918951e+03</td>
      <td>3.335312</td>
      <td>13.300785</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>174</td>
      <td>177</td>
      <td>0.983051</td>
      <td>10.267750</td>
      <td>119.995256</td>
    </tr>
    <tr>
      <th>16</th>
      <td>ENRTER</td>
      <td>2.432581</td>
      <td>49.994997</td>
      <td>53.392460</td>
      <td>1.433107e+02</td>
      <td>110.139221</td>
      <td>45.220661</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>116</td>
      <td>177</td>
      <td>0.655367</td>
      <td>0.275863</td>
      <td>-0.392895</td>
    </tr>
    <tr>
      <th>17</th>
      <td>GDPCAP</td>
      <td>216.827417</td>
      <td>13992.095610</td>
      <td>5348.192875</td>
      <td>1.173705e+05</td>
      <td>51722.069000</td>
      <td>3961.726633</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>170</td>
      <td>177</td>
      <td>0.960452</td>
      <td>2.258568</td>
      <td>5.938690</td>
    </tr>
    <tr>
      <th>18</th>
      <td>EPISCO</td>
      <td>18.900000</td>
      <td>42.946667</td>
      <td>40.900000</td>
      <td>7.790000e+01</td>
      <td>29.600000</td>
      <td>43.600000</td>
      <td>3</td>
      <td>3</td>
      <td>1.000000</td>
      <td>137</td>
      <td>177</td>
      <td>0.774011</td>
      <td>0.641799</td>
      <td>0.035208</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of numeric columns
# with First.Second.Mode.Ratio > 5.00
##################################
len(numeric_column_quality_summary[(numeric_column_quality_summary['First.Second.Mode.Ratio']>5)])
```




    1




```python
##################################
# Identifying the numeric columns
# with First.Second.Mode.Ratio > 5.00
##################################
display(numeric_column_quality_summary[(numeric_column_quality_summary['First.Second.Mode.Ratio']>5)].sort_values(by=['First.Second.Mode.Ratio'], ascending=False))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Minimum</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Maximum</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
      <th>Skewness</th>
      <th>Kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>PM2EXP</td>
      <td>0.274092</td>
      <td>91.940595</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>106</td>
      <td>2</td>
      <td>53.0</td>
      <td>61</td>
      <td>177</td>
      <td>0.344633</td>
      <td>-3.141557</td>
      <td>9.032386</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of numeric columns
# with Unique.Count.Ratio > 10.00
##################################
len(numeric_column_quality_summary[(numeric_column_quality_summary['Unique.Count.Ratio']>10)])
```




    0




```python
##################################
# Counting the number of numeric columns
# with Skewness > 3.00 or Skewness < -3.00
##################################
len(numeric_column_quality_summary[(numeric_column_quality_summary['Skewness']>3) | (numeric_column_quality_summary['Skewness']<(-3))])
```




    5




```python
##################################
# Identifying the numeric columns
# with Skewness > 3.00 or Skewness < -3.00
##################################
display(numeric_column_quality_summary[(numeric_column_quality_summary['Skewness']>3) | (numeric_column_quality_summary['Skewness']<(-3))].sort_values(by=['Skewness'], ascending=False))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Minimum</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Maximum</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
      <th>Skewness</th>
      <th>Kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>POPDEN</td>
      <td>2.115134</td>
      <td>200.886765</td>
      <td>77.983133</td>
      <td>7.918951e+03</td>
      <td>3.335312</td>
      <td>13.300785</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>174</td>
      <td>177</td>
      <td>0.983051</td>
      <td>10.267750</td>
      <td>119.995256</td>
    </tr>
    <tr>
      <th>9</th>
      <td>GHGEMI</td>
      <td>179.725150</td>
      <td>259582.709895</td>
      <td>41009.275980</td>
      <td>1.294287e+07</td>
      <td>571903.119900</td>
      <td>3000.932259</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>170</td>
      <td>177</td>
      <td>0.960452</td>
      <td>9.496120</td>
      <td>101.637308</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PATRES</td>
      <td>1.000000</td>
      <td>20607.388889</td>
      <td>244.500000</td>
      <td>1.344817e+06</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>4</td>
      <td>3</td>
      <td>1.333333</td>
      <td>97</td>
      <td>177</td>
      <td>0.548023</td>
      <td>9.284436</td>
      <td>91.187178</td>
    </tr>
    <tr>
      <th>11</th>
      <td>METEMI</td>
      <td>11.596147</td>
      <td>47876.133575</td>
      <td>11118.976025</td>
      <td>1.186285e+06</td>
      <td>131484.763200</td>
      <td>1326.034028</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>170</td>
      <td>177</td>
      <td>0.960452</td>
      <td>5.801014</td>
      <td>38.661386</td>
    </tr>
    <tr>
      <th>14</th>
      <td>PM2EXP</td>
      <td>0.274092</td>
      <td>91.940595</td>
      <td>100.000000</td>
      <td>1.000000e+02</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>106</td>
      <td>2</td>
      <td>53.000000</td>
      <td>61</td>
      <td>177</td>
      <td>0.344633</td>
      <td>-3.141557</td>
      <td>9.032386</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the dataset
# with object column only
##################################
cancer_rate_object = cancer_rate.select_dtypes(include='object')
```


```python
##################################
# Gathering the variable names for the object column
##################################
object_variable_name_list = cancer_rate_object.columns
```


```python
##################################
# Gathering the first mode values for the object column
##################################
object_first_mode_list = [cancer_rate[x].value_counts().index.tolist()[0] for x in cancer_rate_object]
```


```python
##################################
# Gathering the second mode values for each object column
##################################
object_second_mode_list = [cancer_rate[x].value_counts().index.tolist()[1] for x in cancer_rate_object]
```


```python
##################################
# Gathering the count of first mode values for each object column
##################################
object_first_mode_count_list = [cancer_rate_object[x].isin([cancer_rate[x].value_counts(dropna=True).index.tolist()[0]]).sum() for x in cancer_rate_object]
```


```python
##################################
# Gathering the count of second mode values for each object column
##################################
object_second_mode_count_list = [cancer_rate_object[x].isin([cancer_rate[x].value_counts(dropna=True).index.tolist()[1]]).sum() for x in cancer_rate_object]
```


```python
##################################
# Gathering the first mode to second mode ratio for each object column
##################################
object_first_second_mode_ratio_list = map(truediv, object_first_mode_count_list, object_second_mode_count_list)
```


```python
##################################
# Gathering the count of unique values for each object column
##################################
object_unique_count_list = cancer_rate_object.nunique(dropna=True)
```


```python
##################################
# Gathering the number of observations for each object column
##################################
object_row_count_list = list([len(cancer_rate_object)] * len(cancer_rate_object.columns))
```


```python
##################################
# Gathering the unique to count ratio for each object column
##################################
object_unique_count_ratio_list = map(truediv, object_unique_count_list, object_row_count_list)
```


```python
object_column_quality_summary = pd.DataFrame(zip(object_variable_name_list,
                                                 object_first_mode_list,
                                                 object_second_mode_list,
                                                 object_first_mode_count_list,
                                                 object_second_mode_count_list,
                                                 object_first_second_mode_ratio_list,
                                                 object_unique_count_list,
                                                 object_row_count_list,
                                                 object_unique_count_ratio_list), 
                                        columns=['Object.Column.Name',
                                                 'First.Mode',
                                                 'Second.Mode',
                                                 'First.Mode.Count',
                                                 'Second.Mode.Count',
                                                 'First.Second.Mode.Ratio',
                                                 'Unique.Count',
                                                 'Row.Count',
                                                 'Unique.Count.Ratio'])
display(object_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Object.Column.Name</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>COUNTRY</td>
      <td>Australia</td>
      <td>Mauritius</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>177</td>
      <td>177</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of object columns
# with First.Second.Mode.Ratio > 5.00
##################################
len(object_column_quality_summary[(object_column_quality_summary['First.Second.Mode.Ratio']>5)])
```




    0




```python
##################################
# Counting the number of object columns
# with Unique.Count.Ratio > 10.00
##################################
len(object_column_quality_summary[(object_column_quality_summary['Unique.Count.Ratio']>10)])
```




    0




```python
##################################
# Formulating the dataset
# with categorical columns only
##################################
cancer_rate_categorical = cancer_rate.select_dtypes(include='category')
```


```python
##################################
# Gathering the variable names for the categorical column
##################################
categorical_variable_name_list = cancer_rate_categorical.columns
```


```python
##################################
# Gathering the first mode values for each categorical column
##################################
categorical_first_mode_list = [cancer_rate[x].value_counts().index.tolist()[0] for x in cancer_rate_categorical]
```


```python
##################################
# Gathering the second mode values for each categorical column
##################################
categorical_second_mode_list = [cancer_rate[x].value_counts().index.tolist()[1] for x in cancer_rate_categorical]
```


```python
##################################
# Gathering the count of first mode values for each categorical column
##################################
categorical_first_mode_count_list = [cancer_rate_categorical[x].isin([cancer_rate[x].value_counts(dropna=True).index.tolist()[0]]).sum() for x in cancer_rate_categorical]
```


```python
##################################
# Gathering the count of second mode values for each categorical column
##################################
categorical_second_mode_count_list = [cancer_rate_categorical[x].isin([cancer_rate[x].value_counts(dropna=True).index.tolist()[1]]).sum() for x in cancer_rate_categorical]
```


```python
##################################
# Gathering the first mode to second mode ratio for each categorical column
##################################
categorical_first_second_mode_ratio_list = map(truediv, categorical_first_mode_count_list, categorical_second_mode_count_list)
```


```python
##################################
# Gathering the count of unique values for each categorical column
##################################
categorical_unique_count_list = cancer_rate_categorical.nunique(dropna=True)
```


```python
##################################
# Gathering the number of observations for each categorical column
##################################
categorical_row_count_list = list([len(cancer_rate_categorical)] * len(cancer_rate_categorical.columns))
```


```python
##################################
# Gathering the unique to count ratio for each categorical column
##################################
categorical_unique_count_ratio_list = map(truediv, categorical_unique_count_list, categorical_row_count_list)
```


```python
categorical_column_quality_summary = pd.DataFrame(zip(categorical_variable_name_list,
                                                    categorical_first_mode_list,
                                                    categorical_second_mode_list,
                                                    categorical_first_mode_count_list,
                                                    categorical_second_mode_count_list,
                                                    categorical_first_second_mode_ratio_list,
                                                    categorical_unique_count_list,
                                                    categorical_row_count_list,
                                                    categorical_unique_count_ratio_list), 
                                        columns=['Categorical.Column.Name',
                                                 'First.Mode',
                                                 'Second.Mode',
                                                 'First.Mode.Count',
                                                 'Second.Mode.Count',
                                                 'First.Second.Mode.Ratio',
                                                 'Unique.Count',
                                                 'Row.Count',
                                                 'Unique.Count.Ratio'])
display(categorical_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Categorical.Column.Name</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CANRAT</td>
      <td>Low</td>
      <td>High</td>
      <td>132</td>
      <td>45</td>
      <td>2.933333</td>
      <td>2</td>
      <td>177</td>
      <td>0.011299</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HDICAT</td>
      <td>VH</td>
      <td>H</td>
      <td>59</td>
      <td>39</td>
      <td>1.512821</td>
      <td>4</td>
      <td>177</td>
      <td>0.022599</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of categorical columns
# with First.Second.Mode.Ratio > 5.00
##################################
len(categorical_column_quality_summary[(categorical_column_quality_summary['First.Second.Mode.Ratio']>5)])
```




    0




```python
##################################
# Counting the number of categorical columns
# with Unique.Count.Ratio > 10.00
##################################
len(categorical_column_quality_summary[(categorical_column_quality_summary['Unique.Count.Ratio']>10)])
```




    0



## 1.4. Data Preprocessing <a class="anchor" id="1.4"></a>


### 1.4.1 Data Cleaning <a class="anchor" id="1.4.1"></a>

1. Subsets of rows and columns with high rates of missing data were removed from the dataset:
    * 4 variables with Fill.Rate<0.9 were excluded for subsequent analysis.
        * <span style="color: #FF0000">RNDGDP</span>: Null.Count = 103, Fill.Rate = 0.418
        * <span style="color: #FF0000">PATRES</span>: Null.Count = 69, Fill.Rate = 0.610
        * <span style="color: #FF0000">ENRTER</span>: Null.Count = 61, Fill.Rate = 0.655
        * <span style="color: #FF0000">RELOUT</span>: Null.Count = 24, Fill.Rate = 0.864
    * 14 rows with Missing.Rate>0.2 were exluded for subsequent analysis.
        * <span style="color: #FF0000">COUNTRY=Guadeloupe</span>: Missing.Rate= 0.909
        * <span style="color: #FF0000">COUNTRY=Martinique</span>: Missing.Rate= 0.909
        * <span style="color: #FF0000">COUNTRY=French Guiana</span>: Missing.Rate= 0.909
        * <span style="color: #FF0000">COUNTRY=New Caledonia</span>: Missing.Rate= 0.500
        * <span style="color: #FF0000">COUNTRY=French Polynesia</span>: Missing.Rate= 0.500
        * <span style="color: #FF0000">COUNTRY=Guam</span>: Missing.Rate= 0.500
        * <span style="color: #FF0000">COUNTRY=Puerto Rico</span>: Missing.Rate= 0.409
        * <span style="color: #FF0000">COUNTRY=North Korea</span>: Missing.Rate= 0.227
        * <span style="color: #FF0000">COUNTRY=Somalia</span>: Missing.Rate= 0.227
        * <span style="color: #FF0000">COUNTRY=South Sudan</span>: Missing.Rate= 0.227
        * <span style="color: #FF0000">COUNTRY=Venezuela</span>: Missing.Rate= 0.227
        * <span style="color: #FF0000">COUNTRY=Libya</span>: Missing.Rate= 0.227
        * <span style="color: #FF0000">COUNTRY=Eritrea</span>: Missing.Rate= 0.227
        * <span style="color: #FF0000">COUNTRY=Yemen</span>: Missing.Rate= 0.227  
2. No variables were removed due to zero or near-zero variance.
3. The cleaned dataset is comprised of:
    * **163 rows** (observations)
    * **18 columns** (variables)
        * **1/18 metadata** (object)
            * <span style="color: #FF0000">COUNTRY</span>
        * **1/18 target** (categorical)
             * <span style="color: #FF0000">CANRAT</span>
        * **15/18 predictor** (numeric)
             * <span style="color: #FF0000">GDPPER</span>
             * <span style="color: #FF0000">URBPOP</span>
             * <span style="color: #FF0000">POPGRO</span>
             * <span style="color: #FF0000">LIFEXP</span>
             * <span style="color: #FF0000">TUBINC</span>
             * <span style="color: #FF0000">DTHCMD</span>
             * <span style="color: #FF0000">AGRLND</span>
             * <span style="color: #FF0000">GHGEMI</span>
             * <span style="color: #FF0000">METEMI</span>
             * <span style="color: #FF0000">FORARE</span>
             * <span style="color: #FF0000">CO2EMI</span>
             * <span style="color: #FF0000">PM2EXP</span>
             * <span style="color: #FF0000">POPDEN</span>
             * <span style="color: #FF0000">GDPCAP</span>
             * <span style="color: #FF0000">EPISCO</span>
        * **1/18 predictor** (categorical)
             * <span style="color: #FF0000">HDICAT</span>


```python
##################################
# Performing a general exploration of the original dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate.shape)
```

    Dataset Dimensions: 
    


    (177, 22)



```python
##################################
# Filtering out the rows with
# with Missing.Rate > 0.20
##################################
cancer_rate_filtered_row = cancer_rate.drop(cancer_rate[cancer_rate.COUNTRY.isin(row_high_missing_rate['Row.Name'].values.tolist())].index)
```


```python
##################################
# Performing a general exploration of the filtered dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate_filtered_row.shape)
```

    Dataset Dimensions: 
    


    (163, 22)



```python
##################################
# Filtering out the columns with
# with Fill.Rate < 0.90
##################################
cancer_rate_filtered_row_column = cancer_rate_filtered_row.drop(column_low_fill_rate['Column.Name'].values.tolist(), axis=1)
```


```python
##################################
# Formulating a new dataset object
# for the cleaned data
##################################
cancer_rate_cleaned = cancer_rate_filtered_row_column
```


```python
##################################
# Performing a general exploration of the filtered dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate_cleaned.shape)
```

    Dataset Dimensions: 
    


    (163, 18)


### 1.4.2 Missing Data Imputation <a class="anchor" id="1.4.2"></a>

[Iterative Imputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html) is based on the [Multivariate Imputation by Chained Equations](https://journals.sagepub.com/doi/10.1177/0962280206074463) (MICE) algorithm - an imputation method based on fully conditional specification, where each incomplete variable is imputed by a separate model. As a sequential regression imputation technique, the algorithm imputes an incomplete column (target column) by generating plausible synthetic values given other columns in the data. Each incomplete column must act as a target column, and has its own specific set of predictors. For predictors that are incomplete themselves, the most recently generated imputations are used to complete the predictors prior to prior to imputation of the target columns.

[Linear Regression](https://link.springer.com/book/10.1007/978-1-4757-3462-1) explores the linear relationship between a scalar response and one or more covariates by having the conditional mean of the dependent variable be an affine function of the independent variables. The relationship is modeled through a disturbance term which represents an unobserved random variable that adds noise. The algorithm is typically formulated from the data using the least squares method which seeks to estimate the coefficients by minimizing the squared residual function. The linear equation assigns one scale factor represented by a coefficient to each covariate and an additional coefficient called the intercept or the bias coefficient which gives the line an additional degree of freedom allowing to move up and down a two-dimensional plot.

1. Missing data for numeric variables were imputed using the iterative imputer algorithm with a  linear regression estimator.
    * <span style="color: #FF0000">GDPPER</span>: Null.Count = 1
    * <span style="color: #FF0000">FORARE</span>: Null.Count = 1
    * <span style="color: #FF0000">PM2EXP</span>: Null.Count = 5
2. Missing data for categorical variables were imputed using the most frequent value.
    * <span style="color: #FF0000">HDICAP</span>: Null.Count = 1


```python
##################################
# Formulating the summary
# for all cleaned columns
##################################
cleaned_column_quality_summary = pd.DataFrame(zip(list(cancer_rate_cleaned.columns),
                                                  list(cancer_rate_cleaned.dtypes),
                                                  list([len(cancer_rate_cleaned)] * len(cancer_rate_cleaned.columns)),
                                                  list(cancer_rate_cleaned.count()),
                                                  list(cancer_rate_cleaned.isna().sum(axis=0))), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count'])
display(cleaned_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>COUNTRY</td>
      <td>object</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CANRAT</td>
      <td>category</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GDPPER</td>
      <td>float64</td>
      <td>163</td>
      <td>162</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>URBPOP</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>POPGRO</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LIFEXP</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>TUBINC</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>DTHCMD</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>AGRLND</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>GHGEMI</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>METEMI</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>FORARE</td>
      <td>float64</td>
      <td>163</td>
      <td>162</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>CO2EMI</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>PM2EXP</td>
      <td>float64</td>
      <td>163</td>
      <td>158</td>
      <td>5</td>
    </tr>
    <tr>
      <th>14</th>
      <td>POPDEN</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>GDPCAP</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>HDICAT</td>
      <td>category</td>
      <td>163</td>
      <td>162</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>EPISCO</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the cleaned dataset
# with categorical columns only
##################################
cancer_rate_cleaned_categorical = cancer_rate_cleaned.select_dtypes(include='object')
```


```python
##################################
# Formulating the cleaned dataset
# with numeric columns only
##################################
cancer_rate_cleaned_numeric = cancer_rate_cleaned.select_dtypes(include='number')
```


```python
##################################
# Taking a snapshot of the cleaned dataset
##################################
cancer_rate_cleaned_numeric.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GDPPER</th>
      <th>URBPOP</th>
      <th>POPGRO</th>
      <th>LIFEXP</th>
      <th>TUBINC</th>
      <th>DTHCMD</th>
      <th>AGRLND</th>
      <th>GHGEMI</th>
      <th>METEMI</th>
      <th>FORARE</th>
      <th>CO2EMI</th>
      <th>PM2EXP</th>
      <th>POPDEN</th>
      <th>GDPCAP</th>
      <th>EPISCO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>98380.63601</td>
      <td>86.241</td>
      <td>1.235701</td>
      <td>83.200000</td>
      <td>7.2</td>
      <td>4.941054</td>
      <td>46.252480</td>
      <td>5.719031e+05</td>
      <td>131484.763200</td>
      <td>17.421315</td>
      <td>14.772658</td>
      <td>24.893584</td>
      <td>3.335312</td>
      <td>51722.06900</td>
      <td>60.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>77541.76438</td>
      <td>86.699</td>
      <td>2.204789</td>
      <td>82.256098</td>
      <td>7.2</td>
      <td>4.354730</td>
      <td>38.562911</td>
      <td>8.015803e+04</td>
      <td>32241.937000</td>
      <td>37.570126</td>
      <td>6.160799</td>
      <td>NaN</td>
      <td>19.331586</td>
      <td>41760.59478</td>
      <td>56.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>198405.87500</td>
      <td>63.653</td>
      <td>1.029111</td>
      <td>82.556098</td>
      <td>5.3</td>
      <td>5.684596</td>
      <td>65.495718</td>
      <td>5.949773e+04</td>
      <td>15252.824630</td>
      <td>11.351720</td>
      <td>6.768228</td>
      <td>0.274092</td>
      <td>72.367281</td>
      <td>85420.19086</td>
      <td>57.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>130941.63690</td>
      <td>82.664</td>
      <td>0.964348</td>
      <td>76.980488</td>
      <td>2.3</td>
      <td>5.302060</td>
      <td>44.363367</td>
      <td>5.505181e+06</td>
      <td>748241.402900</td>
      <td>33.866926</td>
      <td>13.032828</td>
      <td>3.343170</td>
      <td>36.240985</td>
      <td>63528.63430</td>
      <td>51.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>113300.60110</td>
      <td>88.116</td>
      <td>0.291641</td>
      <td>81.602439</td>
      <td>4.1</td>
      <td>6.826140</td>
      <td>65.499675</td>
      <td>4.113555e+04</td>
      <td>7778.773921</td>
      <td>15.711000</td>
      <td>4.691237</td>
      <td>56.914456</td>
      <td>145.785100</td>
      <td>60915.42440</td>
      <td>77.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Defining the estimator to be used
# at each step of the round-robin imputation
##################################
lr = LinearRegression()
```


```python
##################################
# Defining the parameter of the
# iterative imputer which will estimate 
# the columns with missing values
# as a function of the other columns
# in a round-robin fashion
##################################
iterative_imputer = IterativeImputer(
    estimator = lr,
    max_iter = 10,
    tol = 1e-10,
    imputation_order = 'ascending',
    random_state=88888888
)
```


```python
##################################
# Implementing the iterative imputer 
##################################
cancer_rate_imputed_numeric_array = iterative_imputer.fit_transform(cancer_rate_cleaned_numeric)
```


```python
##################################
# Transforming the imputed data
# from an array to a dataframe
##################################
cancer_rate_imputed_numeric = pd.DataFrame(cancer_rate_imputed_numeric_array, 
                                           columns = cancer_rate_cleaned_numeric.columns)
```


```python
##################################
# Taking a snapshot of the imputed dataset
##################################
cancer_rate_imputed_numeric.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GDPPER</th>
      <th>URBPOP</th>
      <th>POPGRO</th>
      <th>LIFEXP</th>
      <th>TUBINC</th>
      <th>DTHCMD</th>
      <th>AGRLND</th>
      <th>GHGEMI</th>
      <th>METEMI</th>
      <th>FORARE</th>
      <th>CO2EMI</th>
      <th>PM2EXP</th>
      <th>POPDEN</th>
      <th>GDPCAP</th>
      <th>EPISCO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>98380.63601</td>
      <td>86.241</td>
      <td>1.235701</td>
      <td>83.200000</td>
      <td>7.2</td>
      <td>4.941054</td>
      <td>46.252480</td>
      <td>5.719031e+05</td>
      <td>131484.763200</td>
      <td>17.421315</td>
      <td>14.772658</td>
      <td>24.893584</td>
      <td>3.335312</td>
      <td>51722.06900</td>
      <td>60.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>77541.76438</td>
      <td>86.699</td>
      <td>2.204789</td>
      <td>82.256098</td>
      <td>7.2</td>
      <td>4.354730</td>
      <td>38.562911</td>
      <td>8.015803e+04</td>
      <td>32241.937000</td>
      <td>37.570126</td>
      <td>6.160799</td>
      <td>65.867296</td>
      <td>19.331586</td>
      <td>41760.59478</td>
      <td>56.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>198405.87500</td>
      <td>63.653</td>
      <td>1.029111</td>
      <td>82.556098</td>
      <td>5.3</td>
      <td>5.684596</td>
      <td>65.495718</td>
      <td>5.949773e+04</td>
      <td>15252.824630</td>
      <td>11.351720</td>
      <td>6.768228</td>
      <td>0.274092</td>
      <td>72.367281</td>
      <td>85420.19086</td>
      <td>57.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>130941.63690</td>
      <td>82.664</td>
      <td>0.964348</td>
      <td>76.980488</td>
      <td>2.3</td>
      <td>5.302060</td>
      <td>44.363367</td>
      <td>5.505181e+06</td>
      <td>748241.402900</td>
      <td>33.866926</td>
      <td>13.032828</td>
      <td>3.343170</td>
      <td>36.240985</td>
      <td>63528.63430</td>
      <td>51.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>113300.60110</td>
      <td>88.116</td>
      <td>0.291641</td>
      <td>81.602439</td>
      <td>4.1</td>
      <td>6.826140</td>
      <td>65.499675</td>
      <td>4.113555e+04</td>
      <td>7778.773921</td>
      <td>15.711000</td>
      <td>4.691237</td>
      <td>56.914456</td>
      <td>145.785100</td>
      <td>60915.42440</td>
      <td>77.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Formulating the cleaned dataset
# with categorical columns only
##################################
cancer_rate_cleaned_categorical = cancer_rate_cleaned.select_dtypes(include='category')
```


```python
##################################
# Imputing the missing data
# for categorical columns with
# the most frequent category
##################################
cancer_rate_cleaned_categorical['HDICAT'].fillna(cancer_rate_cleaned_categorical['HDICAT'].mode()[0], inplace=True)
cancer_rate_imputed_categorical = cancer_rate_cleaned_categorical.reset_index(drop=True)
```


```python
##################################
# Formulating the imputed dataset
##################################
cancer_rate_imputed = pd.concat([cancer_rate_imputed_numeric,cancer_rate_imputed_categorical], axis=1, join='inner')  
```


```python
##################################
# Gathering the data types for each column
##################################
data_type_list = list(cancer_rate_imputed.dtypes)
```


```python
##################################
# Gathering the variable names for each column
##################################
variable_name_list = list(cancer_rate_imputed.columns)
```


```python
##################################
# Gathering the number of observations for each column
##################################
row_count_list = list([len(cancer_rate_imputed)] * len(cancer_rate_imputed.columns))
```


```python
##################################
# Gathering the number of missing data for each column
##################################
null_count_list = list(cancer_rate_imputed.isna().sum(axis=0))
```


```python
##################################
# Gathering the number of non-missing data for each column
##################################
non_null_count_list = list(cancer_rate_imputed.count())
```


```python
##################################
# Gathering the missing data percentage for each column
##################################
fill_rate_list = map(truediv, non_null_count_list, row_count_list)
```


```python
##################################
# Formulating the summary
# for all imputed columns
##################################
imputed_column_quality_summary = pd.DataFrame(zip(variable_name_list,
                                                  data_type_list,
                                                  row_count_list,
                                                  non_null_count_list,
                                                  null_count_list,
                                                  fill_rate_list), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count',                                                 
                                                 'Fill.Rate'])
display(imputed_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
      <th>Fill.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GDPPER</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>URBPOP</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>POPGRO</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LIFEXP</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TUBINC</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>DTHCMD</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>AGRLND</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>GHGEMI</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>METEMI</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>FORARE</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>CO2EMI</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>PM2EXP</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>POPDEN</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>GDPCAP</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>EPISCO</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>CANRAT</td>
      <td>category</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>HDICAT</td>
      <td>category</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


### 1.4.3 Outlier Detection <a class="anchor" id="1.4.3"></a>

1. High number of outliers observed for 5 numeric variables with Outlier.Ratio>0.10 and marginal to high Skewness.
    * <span style="color: #FF0000">PM2EXP</span>: Outlier.Count = 37, Outlier.Ratio = 0.226, Skewness=-3.061
    * <span style="color: #FF0000">GHGEMI</span>: Outlier.Count = 27, Outlier.Ratio = 0.165, Skewness=+9.299
    * <span style="color: #FF0000">GDPCAP</span>: Outlier.Count = 22, Outlier.Ratio = 0.134, Skewness=+2.311
    * <span style="color: #FF0000">POPDEN</span>: Outlier.Count = 20, Outlier.Ratio = 0.122, Skewness=+9.972
    * <span style="color: #FF0000">METEMI</span>: Outlier.Count = 20, Outlier.Ratio = 0.122, Skewness=+5.688
2. Minimal number of outliers observed for 5 numeric variables with Outlier.Ratio<0.10 and normal Skewness.
    * <span style="color: #FF0000">TUBINC</span>: Outlier.Count = 12, Outlier.Ratio = 0.073, Skewness=+1.747
    * <span style="color: #FF0000">CO2EMI</span>: Outlier.Count = 11, Outlier.Ratio = 0.067, Skewness=+2.693
    * <span style="color: #FF0000">GDPPER</span>: Outlier.Count = 3, Outlier.Ratio = 0.018, Skewness=+1.554
    * <span style="color: #FF0000">EPISCO</span>: Outlier.Count = 3, Outlier.Ratio = 0.018, Skewness=+0.635
    * <span style="color: #FF0000">CANRAT</span>: Outlier.Count = 2, Outlier.Ratio = 0.012, Skewness=+0.910


```python
##################################
# Formulating the imputed dataset
# with numeric columns only
##################################
cancer_rate_imputed_numeric = cancer_rate_imputed.select_dtypes(include='number')
```


```python
##################################
# Gathering the variable names for each numeric column
##################################
numeric_variable_name_list = list(cancer_rate_imputed_numeric.columns)
```


```python
##################################
# Gathering the skewness value for each numeric column
##################################
numeric_skewness_list = cancer_rate_imputed_numeric.skew()
```


```python
##################################
# Computing the interquartile range
# for all columns
##################################
cancer_rate_imputed_numeric_q1 = cancer_rate_imputed_numeric.quantile(0.25)
cancer_rate_imputed_numeric_q3 = cancer_rate_imputed_numeric.quantile(0.75)
cancer_rate_imputed_numeric_iqr = cancer_rate_imputed_numeric_q3 - cancer_rate_imputed_numeric_q1
```


```python
##################################
# Gathering the outlier count for each numeric column
# based on the interquartile range criterion
##################################
numeric_outlier_count_list = ((cancer_rate_imputed_numeric < (cancer_rate_imputed_numeric_q1 - 1.5 * cancer_rate_imputed_numeric_iqr)) | (cancer_rate_imputed_numeric > (cancer_rate_imputed_numeric_q3 + 1.5 * cancer_rate_imputed_numeric_iqr))).sum()
```


```python
##################################
# Gathering the number of observations for each column
##################################
numeric_row_count_list = list([len(cancer_rate_imputed_numeric)] * len(cancer_rate_imputed_numeric.columns))
```


```python
##################################
# Gathering the unique to count ratio for each categorical column
##################################
numeric_outlier_ratio_list = map(truediv, numeric_outlier_count_list, numeric_row_count_list)
```


```python
##################################
# Formulating the outlier summary
# for all numeric columns
##################################
numeric_column_outlier_summary = pd.DataFrame(zip(numeric_variable_name_list,
                                                  numeric_skewness_list,
                                                  numeric_outlier_count_list,
                                                  numeric_row_count_list,
                                                  numeric_outlier_ratio_list), 
                                        columns=['Numeric.Column.Name',
                                                 'Skewness',
                                                 'Outlier.Count',
                                                 'Row.Count',
                                                 'Outlier.Ratio'])
display(numeric_column_outlier_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Skewness</th>
      <th>Outlier.Count</th>
      <th>Row.Count</th>
      <th>Outlier.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GDPPER</td>
      <td>1.554457</td>
      <td>3</td>
      <td>163</td>
      <td>0.018405</td>
    </tr>
    <tr>
      <th>1</th>
      <td>URBPOP</td>
      <td>-0.212327</td>
      <td>0</td>
      <td>163</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>POPGRO</td>
      <td>-0.181666</td>
      <td>0</td>
      <td>163</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LIFEXP</td>
      <td>-0.329704</td>
      <td>0</td>
      <td>163</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TUBINC</td>
      <td>1.747962</td>
      <td>12</td>
      <td>163</td>
      <td>0.073620</td>
    </tr>
    <tr>
      <th>5</th>
      <td>DTHCMD</td>
      <td>0.930709</td>
      <td>0</td>
      <td>163</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>AGRLND</td>
      <td>0.035315</td>
      <td>0</td>
      <td>163</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>GHGEMI</td>
      <td>9.299960</td>
      <td>27</td>
      <td>163</td>
      <td>0.165644</td>
    </tr>
    <tr>
      <th>8</th>
      <td>METEMI</td>
      <td>5.688689</td>
      <td>20</td>
      <td>163</td>
      <td>0.122699</td>
    </tr>
    <tr>
      <th>9</th>
      <td>FORARE</td>
      <td>0.563015</td>
      <td>0</td>
      <td>163</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>CO2EMI</td>
      <td>2.693585</td>
      <td>11</td>
      <td>163</td>
      <td>0.067485</td>
    </tr>
    <tr>
      <th>11</th>
      <td>PM2EXP</td>
      <td>-3.088403</td>
      <td>37</td>
      <td>163</td>
      <td>0.226994</td>
    </tr>
    <tr>
      <th>12</th>
      <td>POPDEN</td>
      <td>9.972806</td>
      <td>20</td>
      <td>163</td>
      <td>0.122699</td>
    </tr>
    <tr>
      <th>13</th>
      <td>GDPCAP</td>
      <td>2.311079</td>
      <td>22</td>
      <td>163</td>
      <td>0.134969</td>
    </tr>
    <tr>
      <th>14</th>
      <td>EPISCO</td>
      <td>0.635994</td>
      <td>3</td>
      <td>163</td>
      <td>0.018405</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the individual boxplots
# for all numeric columns
##################################
for column in cancer_rate_imputed_numeric:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=cancer_rate_imputed_numeric, x=column)
```


    
![png](output_121_0.png)
    



    
![png](output_121_1.png)
    



    
![png](output_121_2.png)
    



    
![png](output_121_3.png)
    



    
![png](output_121_4.png)
    



    
![png](output_121_5.png)
    



    
![png](output_121_6.png)
    



    
![png](output_121_7.png)
    



    
![png](output_121_8.png)
    



    
![png](output_121_9.png)
    



    
![png](output_121_10.png)
    



    
![png](output_121_11.png)
    



    
![png](output_121_12.png)
    



    
![png](output_121_13.png)
    



    
![png](output_121_14.png)
    


### 1.4.4 Collinearity <a class="anchor" id="1.4.4"></a>

[Pearson’s Correlation Coefficient](https://royalsocietypublishing.org/doi/10.1098/rsta.1896.0007) is a parametric measure of the linear correlation for a pair of features by calculating the ratio between their covariance and the product of their standard deviations. The presence of high absolute correlation values indicate the univariate association between the numeric predictors and the numeric response.

1. Majority of the numeric variables reported moderate to high correlation which were statistically significant.
2. Among pairwise combinations of numeric variables, high Pearson.Correlation.Coefficient values were noted for:
    * <span style="color: #FF0000">GDPPER</span> and <span style="color: #FF0000">GDPCAP</span>: Pearson.Correlation.Coefficient = +0.921
    * <span style="color: #FF0000">GHGEMI</span> and <span style="color: #FF0000">METEMI</span>: Pearson.Correlation.Coefficient = +0.905
3. Among the highly correlated pairs, variables with the lowest correlation against the target variable were removed.
    * <span style="color: #FF0000">GDPPER</span>: Pearson.Correlation.Coefficient = +0.690
    * <span style="color: #FF0000">METEMI</span>: Pearson.Correlation.Coefficient = +0.062
4. The cleaned dataset is comprised of:
    * **163 rows** (observations)
    * **16 columns** (variables)
        * **1/16 metadata** (object)
            * <span style="color: #FF0000">COUNTRY</span>
        * **1/16 target** (categorical)
             * <span style="color: #FF0000">CANRAT</span>
        * **13/16 predictor** (numeric)
             * <span style="color: #FF0000">URBPOP</span>
             * <span style="color: #FF0000">POPGRO</span>
             * <span style="color: #FF0000">LIFEXP</span>
             * <span style="color: #FF0000">TUBINC</span>
             * <span style="color: #FF0000">DTHCMD</span>
             * <span style="color: #FF0000">AGRLND</span>
             * <span style="color: #FF0000">GHGEMI</span>
             * <span style="color: #FF0000">FORARE</span>
             * <span style="color: #FF0000">CO2EMI</span>
             * <span style="color: #FF0000">PM2EXP</span>
             * <span style="color: #FF0000">POPDEN</span>
             * <span style="color: #FF0000">GDPCAP</span>
             * <span style="color: #FF0000">EPISCO</span>
        * **1/16 predictor** (categorical)
             * <span style="color: #FF0000">HDICAT</span>


```python
##################################
# Formulating a function 
# to plot the correlation matrix
# for all pairwise combinations
# of numeric columns
##################################
def plot_correlation_matrix(corr, mask=None):
    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, 
                ax=ax,
                mask=mask,
                annot=True, 
                vmin=-1, 
                vmax=1, 
                center=0,
                cmap='coolwarm', 
                linewidths=1, 
                linecolor='gray', 
                cbar_kws={'orientation': 'horizontal'})  
```


```python
##################################
# Computing the correlation coefficients
# and correlation p-values
# among pairs of numeric columns
##################################
cancer_rate_imputed_numeric_correlation_pairs = {}
cancer_rate_imputed_numeric_columns = cancer_rate_imputed_numeric.columns.tolist()
for numeric_column_a, numeric_column_b in itertools.combinations(cancer_rate_imputed_numeric_columns, 2):
    cancer_rate_imputed_numeric_correlation_pairs[numeric_column_a + '_' + numeric_column_b] = stats.pearsonr(
        cancer_rate_imputed_numeric.loc[:, numeric_column_a], 
        cancer_rate_imputed_numeric.loc[:, numeric_column_b])
```


```python
##################################
# Formulating the pairwise correlation summary
# for all numeric columns
##################################
cancer_rate_imputed_numeric_summary = cancer_rate_imputed_numeric.from_dict(cancer_rate_imputed_numeric_correlation_pairs, orient='index')
cancer_rate_imputed_numeric_summary.columns = ['Pearson.Correlation.Coefficient', 'Correlation.PValue']
display(cancer_rate_imputed_numeric_summary.sort_values(by=['Pearson.Correlation.Coefficient'], ascending=False).head(20))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pearson.Correlation.Coefficient</th>
      <th>Correlation.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GDPPER_GDPCAP</th>
      <td>0.921010</td>
      <td>8.158179e-68</td>
    </tr>
    <tr>
      <th>GHGEMI_METEMI</th>
      <td>0.905121</td>
      <td>1.087643e-61</td>
    </tr>
    <tr>
      <th>POPGRO_DTHCMD</th>
      <td>0.759470</td>
      <td>7.124695e-32</td>
    </tr>
    <tr>
      <th>GDPPER_LIFEXP</th>
      <td>0.755787</td>
      <td>2.055178e-31</td>
    </tr>
    <tr>
      <th>GDPCAP_EPISCO</th>
      <td>0.696707</td>
      <td>5.312642e-25</td>
    </tr>
    <tr>
      <th>LIFEXP_GDPCAP</th>
      <td>0.683834</td>
      <td>8.321371e-24</td>
    </tr>
    <tr>
      <th>GDPPER_EPISCO</th>
      <td>0.680812</td>
      <td>1.555304e-23</td>
    </tr>
    <tr>
      <th>GDPPER_URBPOP</th>
      <td>0.666394</td>
      <td>2.781623e-22</td>
    </tr>
    <tr>
      <th>GDPPER_CO2EMI</th>
      <td>0.654958</td>
      <td>2.450029e-21</td>
    </tr>
    <tr>
      <th>TUBINC_DTHCMD</th>
      <td>0.643615</td>
      <td>1.936081e-20</td>
    </tr>
    <tr>
      <th>URBPOP_LIFEXP</th>
      <td>0.623997</td>
      <td>5.669778e-19</td>
    </tr>
    <tr>
      <th>LIFEXP_EPISCO</th>
      <td>0.620271</td>
      <td>1.048393e-18</td>
    </tr>
    <tr>
      <th>URBPOP_GDPCAP</th>
      <td>0.559181</td>
      <td>8.624533e-15</td>
    </tr>
    <tr>
      <th>CO2EMI_GDPCAP</th>
      <td>0.550221</td>
      <td>2.782997e-14</td>
    </tr>
    <tr>
      <th>URBPOP_CO2EMI</th>
      <td>0.550046</td>
      <td>2.846393e-14</td>
    </tr>
    <tr>
      <th>LIFEXP_CO2EMI</th>
      <td>0.531305</td>
      <td>2.951829e-13</td>
    </tr>
    <tr>
      <th>URBPOP_EPISCO</th>
      <td>0.510131</td>
      <td>3.507463e-12</td>
    </tr>
    <tr>
      <th>POPGRO_TUBINC</th>
      <td>0.442339</td>
      <td>3.384403e-09</td>
    </tr>
    <tr>
      <th>DTHCMD_PM2EXP</th>
      <td>0.283199</td>
      <td>2.491837e-04</td>
    </tr>
    <tr>
      <th>CO2EMI_EPISCO</th>
      <td>0.282734</td>
      <td>2.553620e-04</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting the correlation matrix
# for all pairwise combinations
# of numeric columns
##################################
cancer_rate_imputed_numeric_correlation = cancer_rate_imputed_numeric.corr()
mask = np.triu(cancer_rate_imputed_numeric_correlation)
plot_correlation_matrix(cancer_rate_imputed_numeric_correlation,mask)
plt.show()
```


    
![png](output_126_0.png)
    



```python
##################################
# Formulating a function 
# to plot the correlation matrix
# for all pairwise combinations
# of numeric columns
# with significant p-values only
##################################
def correlation_significance(df=None):
    p_matrix = np.zeros(shape=(df.shape[1],df.shape[1]))
    for col in df.columns:
        for col2 in df.drop(col,axis=1).columns:
            _ , p = stats.pearsonr(df[col],df[col2])
            p_matrix[df.columns.to_list().index(col),df.columns.to_list().index(col2)] = p
    return p_matrix
```


```python
##################################
# Plotting the correlation matrix
# for all pairwise combinations
# of numeric columns
# with significant p-values only
##################################
cancer_rate_imputed_numeric_correlation_p_values = correlation_significance(cancer_rate_imputed_numeric)                     
mask = np.invert(np.tril(cancer_rate_imputed_numeric_correlation_p_values<0.05)) 
plot_correlation_matrix(cancer_rate_imputed_numeric_correlation,mask)  
```


    
![png](output_128_0.png)
    



```python
##################################
# Filtering out one among the 
# highly correlated variable pairs with
# lesser Pearson.Correlation.Coefficient
# when compared to the target variable
##################################
cancer_rate_imputed_numeric.drop(['GDPPER','METEMI'], inplace=True, axis=1)
```


```python
##################################
# Performing a general exploration of the filtered dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate_imputed_numeric.shape)
```

    Dataset Dimensions: 
    


    (163, 13)


### 1.4.5 Shape Transformation <a class="anchor" id="1.4.5"></a>

[Yeo-Johnson Transformation](https://academic.oup.com/biomet/article-abstract/87/4/954/232908?redirectedFrom=fulltext&login=false) applies a new family of distributions that can be used without restrictions, extending many of the good properties of the Box-Cox power family. Similar to the Box-Cox transformation, the method also estimates the optimal value of lambda but has the ability to transform both positive and negative values by inflating low variance data and deflating high variance data to create a more uniform data set. While there are no restrictions in terms of the applicable values, the interpretability of the transformed values is more diminished as compared to the other methods.

1. A Yeo-Johnson transformation was applied to all numeric variables to improve distributional shape.
2. Most variables achieved symmetrical distributions with minimal outliers after transformation.
3. One variable which remained skewed even after applying shape transformation was removed.
    * <span style="color: #FF0000">PM2EXP</span> 
4. The transformed dataset is comprised of:
    * **163 rows** (observations)
    * **15 columns** (variables)
        * **1/15 metadata** (object)
            * <span style="color: #FF0000">COUNTRY</span>
        * **1/15 target** (categorical)
             * <span style="color: #FF0000">CANRAT</span>
        * **12/15 predictor** (numeric)
             * <span style="color: #FF0000">URBPOP</span>
             * <span style="color: #FF0000">POPGRO</span>
             * <span style="color: #FF0000">LIFEXP</span>
             * <span style="color: #FF0000">TUBINC</span>
             * <span style="color: #FF0000">DTHCMD</span>
             * <span style="color: #FF0000">AGRLND</span>
             * <span style="color: #FF0000">GHGEMI</span>
             * <span style="color: #FF0000">FORARE</span>
             * <span style="color: #FF0000">CO2EMI</span>
             * <span style="color: #FF0000">POPDEN</span>
             * <span style="color: #FF0000">GDPCAP</span>
             * <span style="color: #FF0000">EPISCO</span>
        * **1/15 predictor** (categorical)
             * <span style="color: #FF0000">HDICAT</span>


```python
##################################
# Conducting a Yeo-Johnson Transformation
# to address the distributional
# shape of the variables
##################################
yeo_johnson_transformer = PowerTransformer(method='yeo-johnson',
                                          standardize=False)
cancer_rate_imputed_numeric_array = yeo_johnson_transformer.fit_transform(cancer_rate_imputed_numeric)
```


```python
##################################
# Formulating a new dataset object
# for the transformed data
##################################
cancer_rate_transformed_numeric = pd.DataFrame(cancer_rate_imputed_numeric_array,
                                               columns=cancer_rate_imputed_numeric.columns)
```


```python
##################################
# Formulating the individual boxplots
# for all transformed numeric columns
##################################
for column in cancer_rate_transformed_numeric:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=cancer_rate_transformed_numeric, x=column)
```


    
![png](output_134_0.png)
    



    
![png](output_134_1.png)
    



    
![png](output_134_2.png)
    



    
![png](output_134_3.png)
    



    
![png](output_134_4.png)
    



    
![png](output_134_5.png)
    



    
![png](output_134_6.png)
    



    
![png](output_134_7.png)
    



    
![png](output_134_8.png)
    



    
![png](output_134_9.png)
    



    
![png](output_134_10.png)
    



    
![png](output_134_11.png)
    



    
![png](output_134_12.png)
    



```python
##################################
# Filtering out the column
# which remained skewed even
# after applying shape transformation
##################################
cancer_rate_transformed_numeric.drop(['PM2EXP'], inplace=True, axis=1)
```


```python
##################################
# Performing a general exploration of the filtered dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate_transformed_numeric.shape)
```

    Dataset Dimensions: 
    


    (163, 12)


### 1.4.6 Centering and Scaling <a class="anchor" id="1.4.6"></a>

1. All numeric variables were transformed using the standardization method to achieve a comparable scale between values.
4. The scaled dataset is comprised of:
    * **163 rows** (observations)
    * **15 columns** (variables)
        * **1/15 metadata** (object)
            * <span style="color: #FF0000">COUNTRY</span>
        * **1/15 target** (categorical)
             * <span style="color: #FF0000">CANRAT</span>
        * **12/15 predictor** (numeric)
             * <span style="color: #FF0000">URBPOP</span>
             * <span style="color: #FF0000">POPGRO</span>
             * <span style="color: #FF0000">LIFEXP</span>
             * <span style="color: #FF0000">TUBINC</span>
             * <span style="color: #FF0000">DTHCMD</span>
             * <span style="color: #FF0000">AGRLND</span>
             * <span style="color: #FF0000">GHGEMI</span>
             * <span style="color: #FF0000">FORARE</span>
             * <span style="color: #FF0000">CO2EMI</span>
             * <span style="color: #FF0000">POPDEN</span>
             * <span style="color: #FF0000">GDPCAP</span>
             * <span style="color: #FF0000">EPISCO</span>
        * **1/15 predictor** (categorical)
             * <span style="color: #FF0000">HDICAT</span>


```python
##################################
# Conducting standardization
# to transform the values of the 
# variables into comparable scale
##################################
standardization_scaler = StandardScaler()
cancer_rate_transformed_numeric_array = standardization_scaler.fit_transform(cancer_rate_transformed_numeric)
```


```python
##################################
# Formulating a new dataset object
# for the scaled data
##################################
cancer_rate_scaled_numeric = pd.DataFrame(cancer_rate_transformed_numeric_array,
                                          columns=cancer_rate_transformed_numeric.columns)
```


```python
##################################
# Formulating the individual boxplots
# for all transformed numeric columns
##################################
for column in cancer_rate_scaled_numeric:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=cancer_rate_scaled_numeric, x=column)
```


    
![png](output_140_0.png)
    



    
![png](output_140_1.png)
    



    
![png](output_140_2.png)
    



    
![png](output_140_3.png)
    



    
![png](output_140_4.png)
    



    
![png](output_140_5.png)
    



    
![png](output_140_6.png)
    



    
![png](output_140_7.png)
    



    
![png](output_140_8.png)
    



    
![png](output_140_9.png)
    



    
![png](output_140_10.png)
    



    
![png](output_140_11.png)
    


### 1.4.7 Data Encoding <a class="anchor" id="1.4.7"></a>

1. One-hot encoding was applied to the <span style="color: #FF0000">HDICAP_VH</span> variable resulting to 4 additional columns in the dataset:
    * <span style="color: #FF0000">HDICAP_L</span>
    * <span style="color: #FF0000">HDICAP_M</span>
    * <span style="color: #FF0000">HDICAP_H</span>
    * <span style="color: #FF0000">HDICAP_VH</span>


```python
##################################
# Formulating the categorical column
# for encoding transformation
##################################
cancer_rate_categorical_encoded = pd.DataFrame(cancer_rate_cleaned_categorical.loc[:, 'HDICAT'].to_list(),
                                               columns=['HDICAT'])
```


```python
##################################
# Applying a one-hot encoding transformation
# for the categorical column
##################################
cancer_rate_categorical_encoded = pd.get_dummies(cancer_rate_categorical_encoded, columns=['HDICAT'])
```

### 1.4.8 Preprocessed Data Description <a class="anchor" id="1.4.8"></a>

1. The preprocessed dataset is comprised of:
    * **163 rows** (observations)
    * **18 columns** (variables)
        * **1/18 metadata** (object)
            * <span style="color: #FF0000">COUNTRY</span>
        * **1/18 target** (categorical)
             * <span style="color: #FF0000">CANRAT</span>
        * **12/18 predictor** (numeric)
             * <span style="color: #FF0000">URBPOP</span>
             * <span style="color: #FF0000">POPGRO</span>
             * <span style="color: #FF0000">LIFEXP</span>
             * <span style="color: #FF0000">TUBINC</span>
             * <span style="color: #FF0000">DTHCMD</span>
             * <span style="color: #FF0000">AGRLND</span>
             * <span style="color: #FF0000">GHGEMI</span>
             * <span style="color: #FF0000">FORARE</span>
             * <span style="color: #FF0000">CO2EMI</span>
             * <span style="color: #FF0000">POPDEN</span>
             * <span style="color: #FF0000">GDPCAP</span>
             * <span style="color: #FF0000">EPISCO</span>
        * **4/18 predictor** (categorical)
             * <span style="color: #FF0000">HDICAT_L</span>
             * <span style="color: #FF0000">HDICAT_M</span>
             * <span style="color: #FF0000">HDICAT_H</span>
             * <span style="color: #FF0000">HDICAT_VH</span>


```python
##################################
# Consolidating both numeric columns
# and encoded categorical columns
##################################
cancer_rate_preprocessed = pd.concat([cancer_rate_scaled_numeric,cancer_rate_categorical_encoded], axis=1, join='inner')  
```


```python
##################################
# Performing a general exploration of the consolidated dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate_preprocessed.shape)
```

    Dataset Dimensions: 
    


    (163, 16)


## 1.5. Data Exploration <a class="anchor" id="1.5"></a>

### 1.5.1 Exploratory Data Analysis <a class="anchor" id="1.5.1"></a>

1. Bivariate analysis identified individual predictors with generally positive association to the target variable based on visual inspection.
2. Higher values or higher proportions for the following predictors are associated with the <span style="color: #FF0000">CANRAT</span> HIGH category: 
    * <span style="color: #FF0000">URBPOP</span>
    * <span style="color: #FF0000">LIFEXP</span>    
    * <span style="color: #FF0000">CO2EMI</span>    
    * <span style="color: #FF0000">GDPCAP</span>    
    * <span style="color: #FF0000">EPISCO</span>    
    * <span style="color: #FF0000">HDICAP_VH=1</span>
3. Decreasing values or smaller proportions for the following predictors are associated with the <span style="color: #FF0000">CANRAT</span> LOW category: 
    * <span style="color: #FF0000">POPGRO</span>
    * <span style="color: #FF0000">TUBINC</span>    
    * <span style="color: #FF0000">DTHCMD</span> 
    * <span style="color: #FF0000">HDICAP_L=0</span>
    * <span style="color: #FF0000">HDICAP_M=0</span>
    * <span style="color: #FF0000">HDICAP_H=0</span>
4. Values for the following predictors are not associated with the <span style="color: #FF0000">CANRAT</span>  HIGH or LOW categories: 
    * <span style="color: #FF0000">AGRLND</span>
    * <span style="color: #FF0000">GHGEMI</span>    
    * <span style="color: #FF0000">FORARE</span> 
    * <span style="color: #FF0000">POPDEN</span> 


```python
##################################
# Segregating the target
# and predictor variable lists
##################################
cancer_rate_preprocessed_target = cancer_rate_filtered_row['CANRAT'].to_frame()
cancer_rate_preprocessed_target.reset_index(inplace=True, drop=True)
cancer_rate_preprocessed_categorical = cancer_rate_preprocessed[cancer_rate_categorical_encoded.columns]
cancer_rate_preprocessed_categorical_combined = cancer_rate_preprocessed_categorical.join(cancer_rate_preprocessed_target)
cancer_rate_preprocessed = cancer_rate_preprocessed.drop(cancer_rate_categorical_encoded.columns, axis=1) 
cancer_rate_preprocessed_predictors = cancer_rate_preprocessed.columns
cancer_rate_preprocessed_combined = cancer_rate_preprocessed.join(cancer_rate_preprocessed_target)
```


```python
##################################
# Segregating the target
# and predictor variable names
##################################
y_variable = 'CANRAT'
x_variables = cancer_rate_preprocessed_predictors
```


```python
##################################
# Defining the number of 
# rows and columns for the subplots
##################################
num_rows = 6
num_cols = 2
```


```python
##################################
# Formulating the subplot structure
##################################
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 30))

##################################
# Flattening the multi-row and
# multi-column axes
##################################
axes = axes.ravel()

##################################
# Formulating the individual boxplots
# for all scaled numeric columns
##################################
for i, x_variable in enumerate(x_variables):
    ax = axes[i]
    ax.boxplot([group[x_variable] for name, group in cancer_rate_preprocessed_combined.groupby(y_variable)])
    ax.set_title(f'{y_variable} Versus {x_variable}')
    ax.set_xlabel(y_variable)
    ax.set_ylabel(x_variable)
    ax.set_xticks(range(1, len(cancer_rate_preprocessed_combined[y_variable].unique()) + 1), ['Low', 'High'])

##################################
# Adjusting the subplot layout
##################################
plt.tight_layout()

##################################
# Presenting the subplots
##################################
plt.show()
```


    
![png](output_152_0.png)
    



```python
##################################
# Segregating the target
# and predictor variable names
##################################
y_variables = cancer_rate_preprocessed_categorical.columns
x_variable = 'CANRAT'

##################################
# Defining the number of 
# rows and columns for the subplots
##################################
num_rows = 2
num_cols = 2

##################################
# Formulating the subplot structure
##################################
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

##################################
# Flattening the multi-row and
# multi-column axes
##################################
axes = axes.ravel()

##################################
# Formulating the individual stacked column plots
# for all categorical columns
##################################
for i, y_variable in enumerate(y_variables):
    ax = axes[i]
    category_counts = cancer_rate_preprocessed_categorical_combined.groupby([x_variable, y_variable]).size().unstack(fill_value=0)
    category_proportions = category_counts.div(category_counts.sum(axis=1), axis=0)
    category_proportions.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title(f'{x_variable} Versus {y_variable}')
    ax.set_xlabel(x_variable)
    ax.set_ylabel('Proportions')

##################################
# Adjusting the subplot layout
##################################
plt.tight_layout()

##################################
# Presenting the subplots
##################################
plt.show()
```


    
![png](output_153_0.png)
    


### 1.5.2 Hypothesis Testing <a class="anchor" id="1.5.2"></a>

1. The relationship between the numeric predictors to the <span style="color: #FF0000">CANRAT</span> target variable was statistically evaluated using the following hypotheses:
    * **Null**: Difference in the means between groups LOW and HIGH is equal to zero  
    * **Alternative**: Difference in the means between groups LOW and HIGH is not equal to zero   
2. There is sufficient evidence to conclude of a statistically significant difference between the means of the numeric measurements obtained from LOW and HIGH groups of the <span style="color: #FF0000">CANRAT</span> target variable in 9 of the 12 numeric predictors given their high t-test statistic values with reported low p-values less than the significance level of 0.05.
    * <span style="color: #FF0000">GDPCAP</span>: T.Test.Statistic=-11.937, Correlation.PValue=0.000
    * <span style="color: #FF0000">EPISCO</span>: T.Test.Statistic=-11.789, Correlation.PValue=0.000 
    * <span style="color: #FF0000">LIFEXP</span>: T.Test.Statistic=-10.979, Correlation.PValue=0.000  
    * <span style="color: #FF0000">TUBINC</span>: T.Test.Statistic=+9.609, Correlation.PValue=0.000 
    * <span style="color: #FF0000">DTHCMD</span>: T.Test.Statistic=+8.376, Correlation.PValue=0.000 
    * <span style="color: #FF0000">CO2EMI</span>: T.Test.Statistic=-7.031, Correlation.PValue=0.000  
    * <span style="color: #FF0000">URBPOP</span>: T.Test.Statistic=-6.541, Correlation.PValue=0.000   
    * <span style="color: #FF0000">POPGRO</span>: T.Test.Statistic=+4.905, Correlation.PValue=0.000
    * <span style="color: #FF0000">GHGEMI</span>: T.Test.Statistic=-2.243, Correlation.PValue=0.026
3. The relationship between the categorical predictors to the <span style="color: #FF0000">CANRAT</span> target variable was statistically evaluated using the following hypotheses:
    * **Null**: The categorical predictor is independent of the categorical target variable 
    * **Alternative**: The categorical predictor is dependent of the categorical target variable    
2. There is sufficient evidence to conclude of a statistically significant relationship difference between the categories of the categorical predictors and the LOW and HIGH groups of the <span style="color: #FF0000">CANRAT</span> target variable in all 4 categorical predictors given their high chisquare statistic values with reported low p-values less than the significance level of 0.05.
    * <span style="color: #FF0000">HDICAT_VH</span>: ChiSquare.Test.Statistic=76.764, ChiSquare.Test.PValue=0.000
    * <span style="color: #FF0000">HDICAT_H</span>: ChiSquare.Test.Statistic=13.860, ChiSquare.Test.PValue=0.000   
    * <span style="color: #FF0000">HDICAT_M</span>: ChiSquare.Test.Statistic=10.286, ChiSquare.Test.PValue=0.001 
    * <span style="color: #FF0000">HDICAT_L</span>: ChiSquare.Test.Statistic=9.081, ChiSquare.Test.PValue=0.002


```python
##################################
# Computing the t-test 
# statistic and p-values
# between the target variable
# and numeric predictor columns
##################################
cancer_rate_preprocessed_numeric_ttest_target = {}
cancer_rate_preprocessed_numeric = cancer_rate_preprocessed_combined
cancer_rate_preprocessed_numeric_columns = cancer_rate_preprocessed_predictors
for numeric_column in cancer_rate_preprocessed_numeric_columns:
    group_0 = cancer_rate_preprocessed_numeric[cancer_rate_preprocessed_numeric.loc[:,'CANRAT']=='Low']
    group_1 = cancer_rate_preprocessed_numeric[cancer_rate_preprocessed_numeric.loc[:,'CANRAT']=='High']
    cancer_rate_preprocessed_numeric_ttest_target['CANRAT_' + numeric_column] = stats.ttest_ind(
        group_0[numeric_column], 
        group_1[numeric_column], 
        equal_var=True)
```


```python
##################################
# Formulating the pairwise ttest summary
# between the target variable
# and numeric predictor columns
##################################
cancer_rate_preprocessed_numeric_summary = cancer_rate_preprocessed_numeric.from_dict(cancer_rate_preprocessed_numeric_ttest_target, orient='index')
cancer_rate_preprocessed_numeric_summary.columns = ['T.Test.Statistic', 'T.Test.PValue']
display(cancer_rate_preprocessed_numeric_summary.sort_values(by=['T.Test.PValue'], ascending=True).head(12))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>T.Test.Statistic</th>
      <th>T.Test.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CANRAT_GDPCAP</th>
      <td>-11.936988</td>
      <td>6.247937e-24</td>
    </tr>
    <tr>
      <th>CANRAT_EPISCO</th>
      <td>-11.788870</td>
      <td>1.605980e-23</td>
    </tr>
    <tr>
      <th>CANRAT_LIFEXP</th>
      <td>-10.979098</td>
      <td>2.754214e-21</td>
    </tr>
    <tr>
      <th>CANRAT_TUBINC</th>
      <td>9.608760</td>
      <td>1.463678e-17</td>
    </tr>
    <tr>
      <th>CANRAT_DTHCMD</th>
      <td>8.375558</td>
      <td>2.552108e-14</td>
    </tr>
    <tr>
      <th>CANRAT_CO2EMI</th>
      <td>-7.030702</td>
      <td>5.537463e-11</td>
    </tr>
    <tr>
      <th>CANRAT_URBPOP</th>
      <td>-6.541001</td>
      <td>7.734940e-10</td>
    </tr>
    <tr>
      <th>CANRAT_POPGRO</th>
      <td>4.904817</td>
      <td>2.269446e-06</td>
    </tr>
    <tr>
      <th>CANRAT_GHGEMI</th>
      <td>-2.243089</td>
      <td>2.625563e-02</td>
    </tr>
    <tr>
      <th>CANRAT_FORARE</th>
      <td>-1.174143</td>
      <td>2.420717e-01</td>
    </tr>
    <tr>
      <th>CANRAT_POPDEN</th>
      <td>-0.495221</td>
      <td>6.211191e-01</td>
    </tr>
    <tr>
      <th>CANRAT_AGRLND</th>
      <td>-0.047628</td>
      <td>9.620720e-01</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Computing the chisquare
# statistic and p-values
# between the target variable
# and categorical predictor columns
##################################
cancer_rate_preprocessed_categorical_chisquare_target = {}
cancer_rate_preprocessed_categorical = cancer_rate_preprocessed_categorical_combined
cancer_rate_preprocessed_categorical_columns = ['HDICAT_L','HDICAT_M','HDICAT_H','HDICAT_VH']
for categorical_column in cancer_rate_preprocessed_categorical_columns:
    contingency_table = pd.crosstab(cancer_rate_preprocessed_categorical[categorical_column], 
                                    cancer_rate_preprocessed_categorical['CANRAT'])
    cancer_rate_preprocessed_categorical_chisquare_target['CANRAT_' + categorical_column] = stats.chi2_contingency(
        contingency_table)[0:2]
```


```python
##################################
# Formulating the pairwise chisquare summary
# between the target variable
# and categorical predictor columns
##################################
cancer_rate_preprocessed_categorical_summary = cancer_rate_preprocessed_categorical.from_dict(cancer_rate_preprocessed_categorical_chisquare_target, orient='index')
cancer_rate_preprocessed_categorical_summary.columns = ['ChiSquare.Test.Statistic', 'ChiSquare.Test.PValue']
display(cancer_rate_preprocessed_categorical_summary.sort_values(by=['ChiSquare.Test.PValue'], ascending=True).head(4))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ChiSquare.Test.Statistic</th>
      <th>ChiSquare.Test.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CANRAT_HDICAT_VH</th>
      <td>76.764134</td>
      <td>1.926446e-18</td>
    </tr>
    <tr>
      <th>CANRAT_HDICAT_M</th>
      <td>13.860367</td>
      <td>1.969074e-04</td>
    </tr>
    <tr>
      <th>CANRAT_HDICAT_L</th>
      <td>10.285575</td>
      <td>1.340742e-03</td>
    </tr>
    <tr>
      <th>CANRAT_HDICAT_H</th>
      <td>9.080788</td>
      <td>2.583087e-03</td>
    </tr>
  </tbody>
</table>
</div>


## 1.6. Neural Network Classification Gradient and Weight Updates <a class="anchor" id="1.6"></a>

### 1.6.1 Premodelling Data Description <a class="anchor" id="1.6.1"></a>
1. Among the predictor variables determined to have a statistically significant difference between the means of the numeric measurements obtained from LOW and HIGH groups of the <span style="color: #FF0000">CANRAT</span> target variable, only 2 were retained with the highest absolute t-test statistic values with reported low p-values less than the significance level of 0.05.. 
    * <span style="color: #FF0000">GDPCAP</span>: T.Test.Statistic=-11.937, Correlation.PValue=0.000
    * <span style="color: #FF0000">EPISCO</span>: T.Test.Statistic=-11.789, Correlation.PValue=0.000 


```python
##################################
# Filtering certain numeric columns
# and encoded categorical columns
# after hypothesis testing
##################################
cancer_rate_premodelling = cancer_rate_preprocessed_combined.drop(['URBPOP', 'POPGRO', 'LIFEXP', 'TUBINC', 'DTHCMD', 'AGRLND', 'GHGEMI','FORARE', 'CO2EMI', 'POPDEN'], axis=1)
cancer_rate_premodelling.columns
```




    Index(['GDPCAP', 'EPISCO', 'CANRAT'], dtype='object')




```python
##################################
# Performing a general exploration of the filtered dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate_premodelling.shape)
```

    Dataset Dimensions: 
    


    (163, 3)



```python
##################################
# Listing the column names and data types
##################################
print('Column Names and Data Types:')
display(cancer_rate_premodelling.dtypes)
```

    Column Names and Data Types:
    


    GDPCAP     float64
    EPISCO     float64
    CANRAT    category
    dtype: object



```python
##################################
# Taking a snapshot of the dataset
##################################
cancer_rate_premodelling.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GDPCAP</th>
      <th>EPISCO</th>
      <th>CANRAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.549766</td>
      <td>1.306738</td>
      <td>High</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.407752</td>
      <td>1.102912</td>
      <td>High</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.879374</td>
      <td>1.145832</td>
      <td>High</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.685426</td>
      <td>0.739753</td>
      <td>High</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.657777</td>
      <td>2.218327</td>
      <td>High</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Converting the dataframe to
# a numpy array
##################################
cancer_rate_premodelling_matrix = cancer_rate_premodelling.to_numpy()
```


```python
##################################
# Formulating the scatterplot
# of the selected numeric predictors
# by categorical response classes
##################################
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(cancer_rate_premodelling_matrix[cancer_rate_premodelling_matrix[:,2]=='High', 0],
        cancer_rate_premodelling_matrix[cancer_rate_premodelling_matrix[:,2]=='High', 1], 
        'o', 
        label='High', 
        color='darkslateblue')
ax.plot(cancer_rate_premodelling_matrix[cancer_rate_premodelling_matrix[:,2]=='Low', 0],
        cancer_rate_premodelling_matrix[cancer_rate_premodelling_matrix[:,2]=='Low', 1], 
        'x', 
        label='Low', 
        color='chocolate')
ax.axes.set_ylabel('EPISCO')
ax.axes.set_xlabel('GDPCAP')
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.set(title='CANRAT Class Distribution')
ax.legend(loc='upper left',title='CANRAT');
```


    
![png](output_166_0.png)
    



```python
##################################
# Preparing the data and
# and converting to a suitable format
# as a neural network model input
##################################
matrix_x_values = cancer_rate_premodelling.iloc[:,0:2].to_numpy()
y_values = np.where(cancer_rate_premodelling['CANRAT'] == 'High', 1, 0)
```

### 1.6.2 Stochastic Gradient Descent Optimization <a class="anchor" id="1.6.2"></a>

[Backpropagation](https://link.springer.com/book/10.1007/978-0-387-84858-7) and [Weight Update](https://link.springer.com/book/10.1007/978-0-387-84858-7), in the context of an artificial neural network, involve the process of iteratively adjusting the weights of the connections between neurons in the network to minimize the difference between the predicted and the actual target responses. Input data is fed into the neural network, and it propagates through the network layer by layer, starting from the input layer, through hidden layers, and ending at the output layer. At each neuron, the weighted sum of inputs is calculated, followed by the application of an activation function to produce the neuron's output. Once the forward pass is complete, the network's output is compared to the actual target output. The difference between the predicted output and the actual output is quantified using a loss function, which measures the discrepancy between the predicted and actual values. Common loss functions for classification tasks include cross-entropy loss. During the backward pass, the error is propagated backward through the network to compute the gradients of the loss function with respect to each weight in the network. This is achieved using the chain rule of calculus, which allows the error to be decomposed and distributed backward through the network. The gradients quantify how much a change in each weight would affect the overall error of the network. Once the gradients are computed, the weights are updated in the opposite direction of the gradient to minimize the error. This update is typically performed using an optimization algorithm such as gradient descent, which adjusts the weights in proportion to their gradients and a learning rate hyperparameter. The learning rate determines the size of the step taken in the direction opposite to the gradient. These steps are repeated for multiple iterations (epochs) over the training data. As the training progresses, the weights are adjusted iteratively to minimize the error, leading to a neural network model that accurately classifies input data.

[Optimization Algorithms](https://link.springer.com/book/10.1007/978-0-387-84858-7), in the context of neural network classification, are methods used to adjust the parameters (weights and biases) of a neural network during the training process in order to minimize a predefined loss function. The primary goal of these algorithms is to optimize the performance of the neural network by iteratively updating its parameters based on the feedback provided by the training data. Optimization algorithms play a critical role in the training of neural networks because they determine how effectively the network learns from the data and how quickly it converges to an optimal solution. These algorithms are significant during model development in improving model accuracy (optimization algorithms help improve the accuracy of neural network models by minimizing the classification error on the training data), enhancing generalization (by minimizing the loss function during training, optimization algorithms aim to generalize well to unseen data, thereby improving the model's ability to make accurate predictions on new inputs), reducing training time (efficient optimization algorithms can accelerate the convergence of the training process, leading to shorter training times for neural networks), handling complex data (since neural networks often deal with high-dimensional and non-linear data, optimization algorithms enable neural networks to effectively learn complex patterns and relationships within the data, leading to improved classification performance) and adapting to variations in data (optimization algorithms can adapt the model's parameters based on variations in the training data, ensuring robustness and stability in the face of different input distributions or data characteristics).

[Stochastic Gradient Descent Optimization](https://link.springer.com/book/10.1007/978-0-387-84858-7) (SGD) works by iteratively updating the parameters of the neural network in the direction of the negative gradient of the loss function with respect to the parameters. Unlike traditional gradient descent, which computes the gradient using the entire training dataset, SGD computes the gradient using a single randomly selected sample (or a mini-batch of samples) from the dataset. This randomness introduces noise into the gradient estimates but allows SGD to make frequent updates and converge faster. The SGD process involves initializing the parameters of the neural network randomly, shuffling the training dataset and repeating the following steps until convergence - randomly selecting a sample (or a mini-batch of samples) from the dataset, computing the gradient of the loss function with respect to the parameters using the selected sample(s) and updating the parameters using the gradient and a defined learning rate. SGD demonstrates several advantages over other optimization methods in terms of efficiency (SGD is computationally efficient, especially when dealing with large datasets. It updates the parameters using only a subset of the training data in each iteration, making it suitable for training on datasets with millions or even billions of samples.), regularization (SGD introduces noise into the optimization process, which acts as a form of regularization. This helps prevent overfitting and improves the generalization ability of the neural network, especially in situations with limited training data.), and scalability (SGD scales well to deep neural network architectures with a large number of parameters. It can handle complex models with millions of parameters efficiently, making it suitable for modern deep learning applications.). However, some disadvantages of SGD include the variance in gradient estimates (SGD's reliance on single-sample or mini-batch gradient estimates introduces variance into the optimization process. This variance can lead to noisy updates and slow convergence, especially when using small mini-batch sizes.), sensitivity to learning rate (SGD's performance is sensitive to the choice of learning rate. Setting the learning rate too high may lead to unstable updates and divergence, while setting it too low may result in slow convergence and prolonged training times.), and difficulty in choosing learning rate schedule (SGD requires careful tuning of the learning rate schedule to ensure optimal convergence. Finding the right learning rate schedule can be challenging and may require extensive experimentation.)

1. A neural network with the following structure was formulated:
    * <span style="color: #FF0000">Hidden Layer</span> = 3
    * <span style="color: #FF0000">Number of Nodes per Hidden Layer</span> = 5
2. The backpropagation and optimization algorithms were implemented with parameter settings described as follows:
    * <span style="color: #FF0000">Learning Rate</span> = 0.01
    * <span style="color: #FF0000">Epochs</span> = 1000
    * <span style="color: #FF0000">Hidden Layer Activation Function</span> = Sigmoid Activation Function
    * <span style="color: #FF0000">Output Layer Activation Function</span> = Softmax Activation Function
    * <span style="color: #FF0000">Loss Function Optimization Method</span> = Stochastic Gradient Descent (SGD)
3. The final loss estimate determined as 0.18713 at the 1000th epoch was not optimally low as compared to those obtained using the other optimization methods.
4. Applying parameter updates using an SGD cost function optimization, the neural network model performance is estimated as follows:
    * <span style="color: #FF0000">Accuracy</span> = 92.63804
5. The estimated classification accuracy using the SGD cost function optimization was not optimal as compared to those obtained using the other optimization methods. 



```python
##################################
# Defining the neural network architecture
##################################
input_dim = 2
hidden_dims = [5, 5, 5]
output_dim = 2
```


```python
##################################
# Initializing model weights and biases
##################################
params = {}
np.random.seed(88888)
params['W1'] = np.random.randn(input_dim, hidden_dims[0])
params['b1'] = np.zeros(hidden_dims[0])
params['W2'] = np.random.randn(hidden_dims[0], hidden_dims[1])
params['b2'] = np.zeros(hidden_dims[1])
params['W3'] = np.random.randn(hidden_dims[1], hidden_dims[2])
params['b3'] = np.zeros(hidden_dims[2])
params['W4'] = np.random.randn(hidden_dims[2], output_dim)
params['b4'] = np.zeros(output_dim)
```


```python
##################################
# Defining the activation function (ReLU)
##################################
def relu(x):
    return np.maximum(0, x)
```


```python
##################################
# Defining the Softmax function
##################################
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
```


```python
##################################
# Defining the Forward propagation algorithm
##################################
def forward(X, params):
    Z1 = np.dot(X, params['W1']) + params['b1']
    A1 = relu(Z1)
    Z2 = np.dot(A1, params['W2']) + params['b2']
    A2 = relu(Z2)
    Z3 = np.dot(A2, params['W3']) + params['b3']
    A3 = relu(Z3)
    Z4 = np.dot(A3, params['W4']) + params['b4']
    A4 = softmax(Z4)
    return A4, {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3, 'A3': A3, 'Z4': Z4, 'A4': A4}
```


```python
##################################
# Defining the Cross-entropy loss
##################################
def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true])
    loss = np.sum(log_likelihood) / m
    return loss
```


```python
##################################
# Defining the Backpropagation algorithm
##################################
def backward(X, y_true, params, cache):
    m = y_true.shape[0]
    dZ4 = cache['A4'] - np.eye(output_dim)[y_true]
    dW4 = np.dot(cache['A3'].T, dZ4) / m
    db4 = np.sum(dZ4, axis=0) / m
    dA3 = np.dot(dZ4, params['W4'].T)
    dZ3 = dA3 * (cache['Z3'] > 0)
    dW3 = np.dot(cache['A2'].T, dZ3) / m
    db3 = np.sum(dZ3, axis=0) / m
    dA2 = np.dot(dZ3, params['W3'].T)
    dZ2 = dA2 * (cache['Z2'] > 0)
    dW2 = np.dot(cache['A1'].T, dZ2) / m
    db2 = np.sum(dZ2, axis=0) / m
    dA1 = np.dot(dZ2, params['W2'].T)
    dZ1 = dA1 * (cache['Z1'] > 0)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0) / m

    gradients = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3, 'dW4': dW4, 'db4': db4}
    return gradients
```


```python
##################################
# Defining the function to implement
# Stochastic Gradient Descent Optimization
##################################
def sgd(params, gradients, learning_rate):
    for param_name in params:
        params[param_name] -= learning_rate * gradients['d' + param_name]
```


```python
##################################
# Defining the function to implement
# model training
##################################
def train(X, y, params, epochs, learning_rate, optimizer):
    costs = []
    accuracies = []
    for epoch in range(epochs):
        # Performing forward pass
        y_pred, cache = forward(X, params)

        # Computing loss
        loss = cross_entropy_loss(y_pred, y)
        costs.append(loss)

        # Computing accuracy
        accuracy = np.mean(np.argmax(y_pred, axis=1) == y)
        accuracies.append(accuracy)

        # Performing backpropagation
        gradients = backward(X, y, params, cache)

        # Updating the parameters using the specified optimizer
        if optimizer == 'SGD':
            sgd(params, gradients, learning_rate)
        elif optimizer == 'ADAM':
            adam(params, gradients, learning_rate)
        elif optimizer == 'ADAGRAD':
            adagrad(params, gradients, learning_rate)
        elif optimizer == 'ADADELTA':
            adadelta(params, gradients)
        elif optimizer == 'LION':
            lion(params, gradients, learning_rate)
        elif optimizer == 'RMSPROP':
            rmsprop(params, gradients, learning_rate)

        # Printing model iteration progress
        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Loss={loss}, Accuracy={accuracy}')

    return costs, accuracies
```


```python
##################################
# Defining model training parameters
##################################
epochs = 1001
learning_rate = 0.01
```


```python
##################################
# Implementing the method on
# Stochastic Gradient Descent Optimization
##################################
optimizers = ['SGD']
all_costs = {}
all_accuracies = {}
for optimizer in optimizers:
    params_copy = params.copy()
    costs, accuracies = train(matrix_x_values, y_values, params_copy, epochs, learning_rate, optimizer)
    all_costs[optimizer] = costs
    all_accuracies[optimizer] = accuracies
```

    Epoch 0: Loss=0.977656026481093, Accuracy=0.5214723926380368
    Epoch 100: Loss=0.33270037676565895, Accuracy=0.8895705521472392
    Epoch 200: Loss=0.26263234980367245, Accuracy=0.901840490797546
    Epoch 300: Loss=0.23435760093994618, Accuracy=0.901840490797546
    Epoch 400: Loss=0.21806821157745299, Accuracy=0.9079754601226994
    Epoch 500: Loss=0.2074050285456089, Accuracy=0.9263803680981595
    Epoch 600: Loss=0.2001146407114158, Accuracy=0.9263803680981595
    Epoch 700: Loss=0.1949784687584053, Accuracy=0.9263803680981595
    Epoch 800: Loss=0.1914926476761292, Accuracy=0.9263803680981595
    Epoch 900: Loss=0.188994671488959, Accuracy=0.9263803680981595
    Epoch 1000: Loss=0.18713438309683142, Accuracy=0.9263803680981595
    


```python
##################################
# Plotting the cost against iterations for
# Stochastic Gradient Descent Optimization
##################################
plt.figure(figsize=(10, 6))
for optimizer in optimizers:
    plt.plot(range(epochs), all_costs[optimizer], label=optimizer)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('SGD Optimization: Cost Function by Iteration')
plt.ylim(0.15, 0.30)
plt.xlim(-50,1000)
plt.legend([], [], frameon=False)
plt.show()
```


    
![png](output_180_0.png)
    



```python
##################################
# Plotting the classification accuracy against iterations for
# Stochastic Gradient Descent Optimization
##################################
plt.figure(figsize=(10, 6))
for optimizer in optimizers:
    plt.plot(range(epochs), all_accuracies[optimizer], label=optimizer)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('SGD Optimization: : Classification by Iteration')
plt.ylim(0.00, 1.00)
plt.xlim(-50,1000)
plt.legend([], [], frameon=False)
plt.show()
```


    
![png](output_181_0.png)
    



```python
##################################
# Gathering the final accuracy and cost values for 
# Stochastic Gradient Descent Optimization 
##################################
SGD_metrics = pd.DataFrame(["ACCURACY","LOSS"])
SGD_values = pd.DataFrame([accuracies[-1],costs[-1]])
SGD_method = pd.DataFrame(["SGD"]*2)
SGD_summary = pd.concat([SGD_metrics, 
                         SGD_values,
                         SGD_method], axis=1)
SGD_summary.columns = ['Metric', 'Value', 'Method']
SGD_summary.reset_index(inplace=True, drop=True)
display(SGD_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Value</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACCURACY</td>
      <td>0.926380</td>
      <td>SGD</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LOSS</td>
      <td>0.187134</td>
      <td>SGD</td>
    </tr>
  </tbody>
</table>
</div>


### 1.6.3 Adaptive Moment Estimation Optimization <a class="anchor" id="1.6.3"></a>

[Backpropagation](https://link.springer.com/book/10.1007/978-0-387-84858-7) and [Weight Update](https://link.springer.com/book/10.1007/978-0-387-84858-7), in the context of an artificial neural network, involve the process of iteratively adjusting the weights of the connections between neurons in the network to minimize the difference between the predicted and the actual target responses. Input data is fed into the neural network, and it propagates through the network layer by layer, starting from the input layer, through hidden layers, and ending at the output layer. At each neuron, the weighted sum of inputs is calculated, followed by the application of an activation function to produce the neuron's output. Once the forward pass is complete, the network's output is compared to the actual target output. The difference between the predicted output and the actual output is quantified using a loss function, which measures the discrepancy between the predicted and actual values. Common loss functions for classification tasks include cross-entropy loss. During the backward pass, the error is propagated backward through the network to compute the gradients of the loss function with respect to each weight in the network. This is achieved using the chain rule of calculus, which allows the error to be decomposed and distributed backward through the network. The gradients quantify how much a change in each weight would affect the overall error of the network. Once the gradients are computed, the weights are updated in the opposite direction of the gradient to minimize the error. This update is typically performed using an optimization algorithm such as gradient descent, which adjusts the weights in proportion to their gradients and a learning rate hyperparameter. The learning rate determines the size of the step taken in the direction opposite to the gradient. These steps are repeated for multiple iterations (epochs) over the training data. As the training progresses, the weights are adjusted iteratively to minimize the error, leading to a neural network model that accurately classifies input data. This activation function is commonly used in the output layer for binary classification problems.

[Optimization Algorithms](https://link.springer.com/book/10.1007/978-0-387-84858-7), in the context of neural network classification, are methods used to adjust the parameters (weights and biases) of a neural network during the training process in order to minimize a predefined loss function. The primary goal of these algorithms is to optimize the performance of the neural network by iteratively updating its parameters based on the feedback provided by the training data. Optimization algorithms play a critical role in the training of neural networks because they determine how effectively the network learns from the data and how quickly it converges to an optimal solution. These algorithms are significant during model development in improving model accuracy (optimization algorithms help improve the accuracy of neural network models by minimizing the classification error on the training data), enhancing generalization (by minimizing the loss function during training, optimization algorithms aim to generalize well to unseen data, thereby improving the model's ability to make accurate predictions on new inputs), reducing training time (efficient optimization algorithms can accelerate the convergence of the training process, leading to shorter training times for neural networks), handling complex data (since neural networks often deal with high-dimensional and non-linear data, optimization algorithms enable neural networks to effectively learn complex patterns and relationships within the data, leading to improved classification performance) and adapting to variations in data (optimization algorithms can adapt the model's parameters based on variations in the training data, ensuring robustness and stability in the face of different input distributions or data characteristics).

[Adaptive Moment Estimation Optimization](https://link.springer.com/book/10.1007/978-0-387-84858-7) (Adam) combines both momentum-based methods and adaptive learning rate methods by maintaining exponentially decaying moving averages of past gradients and their squares, which are then used to adaptively adjust the learning rates for each parameter. The Adam process involves initializing the parameters, including the first and second moment estimates (m and v) to zero. In each iteration of training which is repeated until convergence or a predetermined number of iterations, the gradients of the loss function with respect to the parameters are computed, the biased first and second moment estimates are sequentially determined, the bias in the first and second moment estimates are corrected and the model parameters are subsequently updated. Adam demonstrates several advantages over other optimization methods in terms of adaptive learning rates (Adam adapts the learning rates for each parameter individually, making it less sensitive to manual tuning of learning rate hyperparameters compared to SGD and RMSprop), efficient convergence (Adam often converges faster than SGD and RMSprop, especially in the presence of sparse gradients or non-stationary objectives.), and robustness to noisy gradients (Adam's adaptive learning rate mechanism and momentum-like updates make it more robust to noisy gradients compared to SGD and AdaGrad.). However, some disadvantages of Adam include memory and computational cost (Adam requires additional memory and computation to maintain the moving average estimates of the gradients and their squares. This can increase the computational overhead, especially for large-scale neural networks.), sensitivity to hyperparameters (although Adam is less sensitive to learning rate hyperparameters compared to SGD, it still requires tuning of other hyperparameters such as the momentum parameters, and potential overfitting (In some cases, Adam may exhibit aggressive updates, leading to overfitting, especially when the momentum parameters are not properly tuned.).

1. A neural network with the following structure was formulated:
    * <span style="color: #FF0000">Hidden Layer</span> = 3
    * <span style="color: #FF0000">Number of Nodes per Hidden Layer</span> = 5
2. The backpropagation and optimization algorithms were implemented with parameter settings described as follows:
    * <span style="color: #FF0000">Learning Rate</span> = 0.01
    * <span style="color: #FF0000">Epochs</span> = 1000
    * <span style="color: #FF0000">Hidden Layer Activation Function</span> = Sigmoid Activation Function
    * <span style="color: #FF0000">Output Layer Activation Function</span> = Softmax Activation Function
    * <span style="color: #FF0000">Loss Function Optimization Method</span> = Adaptive Moment Estimation (ADAM)
    * <span style="color: #FF0000">Beta1 (Exponential Decay Rate for First Moment)</span> = 0.900
    * <span style="color: #FF0000">Beta2 (Exponential Decay Rate for Second Moment)</span> = 0.999
    * <span style="color: #FF0000">Epsilon (Constant to Maintain Numerical Stability During Update)</span> = 1e-8
3. The final loss estimate determined as 0.17663 at the 1000th epoch was not optimally low as compared to those obtained using the other optimization methods.
4. Applying parameter updates using an ADAM cost function optimization, the neural network model performance is estimated as follows:
    * <span style="color: #FF0000">Accuracy</span> = 91.41104
5. The estimated classification accuracy using the ADAM cost function optimization was not optimal as compared to those obtained using the other optimization methods. 



```python
##################################
# Defining the function to implement
# Adaptive Moment Estimation Optimization
##################################
def adam(params, gradients, learning_rate, m=None, v=None, beta1=0.9, beta2=0.999, eps=1e-8, t=0):
    if m is None:
        m = {k: np.zeros_like(v) for k, v in params.items()}
    if v is None:
        v = {k: np.zeros_like(v) for k, v in params.items()}

    t += 1
    for param_name in params:
        m[param_name] = beta1 * m[param_name] + (1 - beta1) * gradients['d' + param_name]
        v[param_name] = beta2 * v[param_name] + (1 - beta2) * (gradients['d' + param_name] ** 2)
        m_hat = m[param_name] / (1 - beta1 ** t)
        v_hat = v[param_name] / (1 - beta2 ** t)
        params[param_name] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)
```


```python
##################################
# Defining model training parameters
##################################
epochs = 1001
learning_rate = 0.01
```


```python
##################################
# Implementing the method on
# Adaptive Moment Estimation Optimization
##################################
optimizers = ['ADAM']
all_costs = {}
all_accuracies = {}
for optimizer in optimizers:
    params_copy = params.copy()
    costs, accuracies = train(matrix_x_values, y_values, params_copy, epochs, learning_rate, optimizer)
    all_costs[optimizer] = costs
    all_accuracies[optimizer] = accuracies
```

    Epoch 0: Loss=0.18711788665783202, Accuracy=0.9263803680981595
    Epoch 100: Loss=0.18001254333263383, Accuracy=0.9141104294478528
    Epoch 200: Loss=0.17884340079313124, Accuracy=0.9141104294478528
    Epoch 300: Loss=0.17809955980988298, Accuracy=0.9141104294478528
    Epoch 400: Loss=0.17764474833280822, Accuracy=0.9141104294478528
    Epoch 500: Loss=0.17731162147743887, Accuracy=0.9141104294478528
    Epoch 600: Loss=0.17706218501973514, Accuracy=0.9141104294478528
    Epoch 700: Loss=0.17687868107722898, Accuracy=0.9141104294478528
    Epoch 800: Loss=0.17674860655817856, Accuracy=0.9141104294478528
    Epoch 900: Loss=0.17669745796295797, Accuracy=0.9141104294478528
    Epoch 1000: Loss=0.17663295512555588, Accuracy=0.9141104294478528
    


```python
##################################
# Plotting the cost against iterations for
# Adaptive Moment Estimation Optimization
##################################
plt.figure(figsize=(10, 6))
for optimizer in optimizers:
    plt.plot(range(epochs), all_costs[optimizer], label=optimizer)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('ADAM Optimization: Cost Function by Iteration')
plt.ylim(0.15, 0.30)
plt.xlim(-50,1000)
plt.legend([], [], frameon=False)
plt.show()
```


    
![png](output_187_0.png)
    



```python
##################################
# Plotting the classification accuracy against iterations for
# Adaptive Moment Estimation Optimization
##################################
plt.figure(figsize=(10, 6))
for optimizer in optimizers:
    plt.plot(range(epochs), all_accuracies[optimizer], label=optimizer)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('ADAM Optimization: : Classification by Iteration')
plt.ylim(0.00, 1.00)
plt.xlim(-50,1000)
plt.legend([], [], frameon=False)
plt.show()
```


    
![png](output_188_0.png)
    



```python
##################################
# Gathering the final accuracy and cost values for 
# Adaptive Moment Estimation Optimization
##################################
ADAM_metrics = pd.DataFrame(["ACCURACY","LOSS"])
ADAM_values = pd.DataFrame([accuracies[-1],costs[-1]])
ADAM_method = pd.DataFrame(["ADAM"]*2)
ADAM_summary = pd.concat([ADAM_metrics, 
                         ADAM_values,
                         ADAM_method], axis=1)
ADAM_summary.columns = ['Metric', 'Value', 'Method']
ADAM_summary.reset_index(inplace=True, drop=True)
display(ADAM_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Value</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACCURACY</td>
      <td>0.914110</td>
      <td>ADAM</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LOSS</td>
      <td>0.176633</td>
      <td>ADAM</td>
    </tr>
  </tbody>
</table>
</div>


### 1.6.4 Adaptive Gradient Algorithm Optimization <a class="anchor" id="1.6.4"></a>

[Backpropagation](https://link.springer.com/book/10.1007/978-0-387-84858-7) and [Weight Update](https://link.springer.com/book/10.1007/978-0-387-84858-7), in the context of an artificial neural network, involve the process of iteratively adjusting the weights of the connections between neurons in the network to minimize the difference between the predicted and the actual target responses. Input data is fed into the neural network, and it propagates through the network layer by layer, starting from the input layer, through hidden layers, and ending at the output layer. At each neuron, the weighted sum of inputs is calculated, followed by the application of an activation function to produce the neuron's output. Once the forward pass is complete, the network's output is compared to the actual target output. The difference between the predicted output and the actual output is quantified using a loss function, which measures the discrepancy between the predicted and actual values. Common loss functions for classification tasks include cross-entropy loss. During the backward pass, the error is propagated backward through the network to compute the gradients of the loss function with respect to each weight in the network. This is achieved using the chain rule of calculus, which allows the error to be decomposed and distributed backward through the network. The gradients quantify how much a change in each weight would affect the overall error of the network. Once the gradients are computed, the weights are updated in the opposite direction of the gradient to minimize the error. This update is typically performed using an optimization algorithm such as gradient descent, which adjusts the weights in proportion to their gradients and a learning rate hyperparameter. The learning rate determines the size of the step taken in the direction opposite to the gradient. These steps are repeated for multiple iterations (epochs) over the training data. As the training progresses, the weights are adjusted iteratively to minimize the error, leading to a neural network model that accurately classifies input data.

[Optimization Algorithms](https://link.springer.com/book/10.1007/978-0-387-84858-7), in the context of neural network classification, are methods used to adjust the parameters (weights and biases) of a neural network during the training process in order to minimize a predefined loss function. The primary goal of these algorithms is to optimize the performance of the neural network by iteratively updating its parameters based on the feedback provided by the training data. Optimization algorithms play a critical role in the training of neural networks because they determine how effectively the network learns from the data and how quickly it converges to an optimal solution. These algorithms are significant during model development in improving model accuracy (optimization algorithms help improve the accuracy of neural network models by minimizing the classification error on the training data), enhancing generalization (by minimizing the loss function during training, optimization algorithms aim to generalize well to unseen data, thereby improving the model's ability to make accurate predictions on new inputs), reducing training time (efficient optimization algorithms can accelerate the convergence of the training process, leading to shorter training times for neural networks), handling complex data (since neural networks often deal with high-dimensional and non-linear data, optimization algorithms enable neural networks to effectively learn complex patterns and relationships within the data, leading to improved classification performance) and adapting to variations in data (optimization algorithms can adapt the model's parameters based on variations in the training data, ensuring robustness and stability in the face of different input distributions or data characteristics).

[Adaptive Gradient Algorithm Optimization](https://link.springer.com/book/10.1007/978-0-387-84858-7) (AdaGrad) adapts the learning rates of individual parameters based on the historical gradient information. The main idea behind AdaGrad is to decrease the learning rate for parameters that have been updated frequently and increase the learning rate for parameters that have been updated infrequently. The AdaGrad process involves initializing the parameters, including the squared gradient accumulation variable (denoted as "cache"), to a small positive value. In each iteration of training which is repeated until convergence or a predetermined number of iterations, the gradients of the loss function with respect to the parameters are computed, the squared gradient accumulation variable is determined, and the parameters are subsequently updated using the accumulated squared gradients. AdaGrad demonstrates several advantages over other optimization methods in terms of adaptive learning rates (AdaGrad adapts the learning rates for each parameter individually, making it less sensitive to manual tuning of learning rate hyperparameters compared to SGD.), efficient handling of sparse data (AdaGrad performs well in scenarios where the data is sparse or features have varying importance. It adjusts the learning rates based on the accumulated gradients, which helps handle such data efficiently.), and quick convergence (AdaGrad often converges quickly, especially in settings where the learning rates need to be adjusted dynamically based on the gradients' characteristics. This efficiency can lead to faster convergence compared to SGD.). However, some disadvantages of AdaGrad include diminishing learning rates (AdaGrad's accumulation of squared gradients can lead to diminishing learning rates over time. As the accumulation increases, the learning rates for parameters may become very small, which can slow down the learning process, especially in later stages of training.), memory and computational cost (AdaGrad requires additional memory and computation to store and update the accumulated squared gradients for each parameter. This overhead can become significant for large-scale neural networks with a high number of parameters.), and potential oversensitivity to initial learning rate (AdaGrad's performance can be sensitive to the initial learning rate setting. If the initial learning rate is too high, AdaGrad may converge prematurely, while a too low initial learning rate can lead to slow convergence.).

1. A neural network with the following structure was formulated:
    * <span style="color: #FF0000">Hidden Layer</span> = 3
    * <span style="color: #FF0000">Number of Nodes per Hidden Layer</span> = 5
2. The backpropagation and optimization algorithms were implemented with parameter settings described as follows:
    * <span style="color: #FF0000">Learning Rate</span> = 0.01
    * <span style="color: #FF0000">Epochs</span> = 1000
    * <span style="color: #FF0000">Hidden Layer Activation Function</span> = Sigmoid Activation Function
    * <span style="color: #FF0000">Output Layer Activation Function</span> = Softmax Activation Function
    * <span style="color: #FF0000">Loss Function Optimization Method</span> = Adaptive Gradient Algorithm (ADAGRAD)
    * <span style="color: #FF0000">Epsilon (Constant to Maintain Numerical Stability During Update)</span> = 1e-8
3. The final loss estimate determined as 0.17211 at the 1000th epoch was not optimally low as compared to those obtained using the other optimization methods.
4. Applying parameter updates using an ADAGRAD cost function optimization, the neural network model performance is estimated as follows:
    * <span style="color: #FF0000">Accuracy</span> = 92.63803
5. The estimated classification accuracy using the ADAGRAD cost function optimization was not optimal as compared to those obtained using the other optimization methods. 



```python
##################################
# Defining the function to implement
# Adaptive Gradient Algorithm Optimization
##################################
def adagrad(params, gradients, learning_rate, cache=None, eps=1e-8):
    if cache is None:
        cache = {key: np.zeros_like(value) for key, value in params.items()}
    
    for key in params.keys():
        cache[key] += gradients['d' + key] ** 2
        params[key] -= learning_rate * gradients['d' + key] / (np.sqrt(cache[key]) + eps)   
```


```python
##################################
# Defining model training parameters
##################################
epochs = 1001
learning_rate = 0.01
```


```python
##################################
# Implementing the method on
# Adaptive Gradient Algorithm Optimization
##################################
optimizers = ['ADAGRAD']
all_costs = {}
all_accuracies = {}
for optimizer in optimizers:
    params_copy = params.copy()
    costs, accuracies = train(matrix_x_values, y_values, params_copy, epochs, learning_rate, optimizer)
    all_costs[optimizer] = costs
    all_accuracies[optimizer] = accuracies
```

    Epoch 0: Loss=0.17255033561236358, Accuracy=0.9263803680981595
    Epoch 100: Loss=0.17225880365181617, Accuracy=0.9263803680981595
    Epoch 200: Loss=0.1721480527654002, Accuracy=0.9263803680981595
    Epoch 300: Loss=0.17214368549141815, Accuracy=0.9263803680981595
    Epoch 400: Loss=0.17213930321391052, Accuracy=0.9263803680981595
    Epoch 500: Loss=0.17213491306759807, Accuracy=0.9263803680981595
    Epoch 600: Loss=0.17213051665091217, Accuracy=0.9263803680981595
    Epoch 700: Loss=0.17212611430185815, Accuracy=0.9263803680981595
    Epoch 800: Loss=0.17212170607483865, Accuracy=0.9263803680981595
    Epoch 900: Loss=0.17211729196073022, Accuracy=0.9263803680981595
    Epoch 1000: Loss=0.17211287193611693, Accuracy=0.9263803680981595
    


```python
##################################
# Plotting the cost against iterations for
# Adaptive Gradient Algorithm Optimization
##################################
plt.figure(figsize=(10, 6))
for optimizer in optimizers:
    plt.plot(range(epochs), all_costs[optimizer], label=optimizer)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('ADAGRAD Optimization: Cost Function by Iteration')
plt.ylim(0.15, 0.30)
plt.xlim(-50,1000)
plt.legend([], [], frameon=False)
plt.show()
```


    
![png](output_194_0.png)
    



```python
##################################
# Plotting the classification accuracy against iterations for
# Adaptive Gradient Algorithm Optimization
##################################
plt.figure(figsize=(10, 6))
for optimizer in optimizers:
    plt.plot(range(epochs), all_accuracies[optimizer], label=optimizer)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('ADAGRAD Optimization: : Classification by Iteration')
plt.ylim(0.00, 1.00)
plt.xlim(-50,1000)
plt.legend([], [], frameon=False)
plt.show()
```


    
![png](output_195_0.png)
    



```python
##################################
# Gathering the final accuracy and cost values for 
# Adaptive Gradient Algorithm Optimization
##################################
ADAGRAD_metrics = pd.DataFrame(["ACCURACY","LOSS"])
ADAGRAD_values = pd.DataFrame([accuracies[-1],costs[-1]])
ADAGRAD_method = pd.DataFrame(["ADAGRAD"]*2)
ADAGRAD_summary = pd.concat([ADAGRAD_metrics, 
                         ADAGRAD_values,
                         ADAGRAD_method], axis=1)
ADAGRAD_summary.columns = ['Metric', 'Value', 'Method']
ADAGRAD_summary.reset_index(inplace=True, drop=True)
display(ADAGRAD_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Value</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACCURACY</td>
      <td>0.926380</td>
      <td>ADAGRAD</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LOSS</td>
      <td>0.172113</td>
      <td>ADAGRAD</td>
    </tr>
  </tbody>
</table>
</div>


### 1.6.5 AdaDelta Optimization <a class="anchor" id="1.6.5"></a>

[Backpropagation](https://link.springer.com/book/10.1007/978-0-387-84858-7) and [Weight Update](https://link.springer.com/book/10.1007/978-0-387-84858-7), in the context of an artificial neural network, involve the process of iteratively adjusting the weights of the connections between neurons in the network to minimize the difference between the predicted and the actual target responses. Input data is fed into the neural network, and it propagates through the network layer by layer, starting from the input layer, through hidden layers, and ending at the output layer. At each neuron, the weighted sum of inputs is calculated, followed by the application of an activation function to produce the neuron's output. Once the forward pass is complete, the network's output is compared to the actual target output. The difference between the predicted output and the actual output is quantified using a loss function, which measures the discrepancy between the predicted and actual values. Common loss functions for classification tasks include cross-entropy loss. During the backward pass, the error is propagated backward through the network to compute the gradients of the loss function with respect to each weight in the network. This is achieved using the chain rule of calculus, which allows the error to be decomposed and distributed backward through the network. The gradients quantify how much a change in each weight would affect the overall error of the network. Once the gradients are computed, the weights are updated in the opposite direction of the gradient to minimize the error. This update is typically performed using an optimization algorithm such as gradient descent, which adjusts the weights in proportion to their gradients and a learning rate hyperparameter. The learning rate determines the size of the step taken in the direction opposite to the gradient. These steps are repeated for multiple iterations (epochs) over the training data. As the training progresses, the weights are adjusted iteratively to minimize the error, leading to a neural network model that accurately classifies input data.

[Optimization Algorithms](https://link.springer.com/book/10.1007/978-0-387-84858-7), in the context of neural network classification, are methods used to adjust the parameters (weights and biases) of a neural network during the training process in order to minimize a predefined loss function. The primary goal of these algorithms is to optimize the performance of the neural network by iteratively updating its parameters based on the feedback provided by the training data. Optimization algorithms play a critical role in the training of neural networks because they determine how effectively the network learns from the data and how quickly it converges to an optimal solution. These algorithms are significant during model development in improving model accuracy (optimization algorithms help improve the accuracy of neural network models by minimizing the classification error on the training data), enhancing generalization (by minimizing the loss function during training, optimization algorithms aim to generalize well to unseen data, thereby improving the model's ability to make accurate predictions on new inputs), reducing training time (efficient optimization algorithms can accelerate the convergence of the training process, leading to shorter training times for neural networks), handling complex data (since neural networks often deal with high-dimensional and non-linear data, optimization algorithms enable neural networks to effectively learn complex patterns and relationships within the data, leading to improved classification performance) and adapting to variations in data (optimization algorithms can adapt the model's parameters based on variations in the training data, ensuring robustness and stability in the face of different input distributions or data characteristics).

[AdaDelta Optimization](https://link.springer.com/book/10.1007/978-0-387-84858-7) (AdaDelta) is an extension of AdaGrad and addresses its limitation of diminishing learning rates over time. AdaDelta dynamically adapts the learning rates based on a moving average of past gradients and updates without the need for an explicit learning rate parameter. The AdaDelta process involves initializing the parameters, including the moving average variables for the gradient and the parameter update to zero, and setting a decay rate parameter. In each iteration of training which is repeated until convergence or a predetermined number of iterations, the gradients of the loss function with respect to the parameters are computed, the moving average variables are estimated, the parameter updates based on a moving average of past updates are determined, and the model parameters with their associated moving averages are calculated. AdaDelta demonstrates several advantages over other optimization methods in terms of no manual learning rate tuning (AdaDelta eliminates the need for manually tuning learning rate hyperparameters, making it more user-friendly and robust to variations in data and architectures compared to methods like SGD and Adam.), efficient handling of sparse gradients (AdaDelta performs well in scenarios where the gradients are sparse or have varying magnitudes. Its adaptive learning rate mechanism allows it to handle such gradients efficiently, leading to improved optimization performance.), and no diminishing learning rates (AdaDelta addresses the issue of diminishing learning rates over time, which can occur in AdaGrad and RMSprop. By incorporating a moving average of past updates, AdaDelta ensures that the learning rates remain relevant throughout the training process.). However, some disadvantages of AdaDelta include memory and computational cost (AdaDelta requires additional memory and computation to store and update the moving averages of gradients and updates. This overhead can become significant for large-scale neural networks with a high number of parameters.), sensitivity to decay rate parameter (the performance of AdaDelta can be sensitive to the choice of the decay rate parameter. Setting this parameter too low may result in slow convergence, while setting it too high may lead to instability or oscillations in the optimization process.), and potential overshooting (AdaDelta's reliance on a moving average of past updates may lead to overshooting or oscillations in the optimization process, especially in scenarios with highly non-convex objectives or noisy gradients.).

1. A neural network with the following structure was formulated:
    * <span style="color: #FF0000">Hidden Layer</span> = 3
    * <span style="color: #FF0000">Number of Nodes per Hidden Layer</span> = 5
2. The backpropagation and optimization algorithms were implemented with parameter settings described as follows:
    * <span style="color: #FF0000">Learning Rate</span> = 0.01
    * <span style="color: #FF0000">Epochs</span> = 1000
    * <span style="color: #FF0000">Hidden Layer Activation Function</span> = Sigmoid Activation Function
    * <span style="color: #FF0000">Output Layer Activation Function</span> = Softmax Activation Function
    * <span style="color: #FF0000">Loss Function Optimization Method</span> = AdaDelta Optimization (ADADELTA)
    * <span style="color: #FF0000">Rho (Exponential Decay of Accumulated Past Gradients)</span> = 0.900
    * <span style="color: #FF0000">Epsilon (Constant to Maintain Numerical Stability During Update)</span> = 1e-8
3. The final loss estimate determined as 0.15831 at the 1000th epoch was not optimally low as compared to those obtained using the other optimization methods.
4. Applying parameter updates using an ADADELTA cost function optimization, the neural network model performance is estimated as follows:
    * <span style="color: #FF0000">Accuracy</span> = 92.63803
5. The estimated classification accuracy using the ADADELTA cost function optimization was not optimal as compared to those obtained using the other optimization methods. 



```python
##################################
# Defining the function to implement
# AdaDelta Optimization 
##################################
def adadelta(params, gradients, cache=None, rho=0.9, eps=1e-8):
    if cache is None:
        cache = {key: np.zeros_like(value) for key, value in params.items()}
    delta = {key: np.zeros_like(value) for key, value in params.items()}
    
    for key in params.keys():
        cache[key] = rho * cache[key] + (1 - rho) * (gradients['d' + key] ** 2)
        delta[key] = -np.sqrt(delta[key] + eps) * gradients['d' + key] / np.sqrt(cache[key] + eps)
        params[key] += delta[key]
        delta[key] = rho * delta[key] + (1 - rho) * (delta[key] ** 2)
```


```python
##################################
# Defining model training parameters
##################################
epochs = 1001
learning_rate = 0.01
```


```python
##################################
# Implementing the method on
# AdaDelta Optimization
##################################
optimizers = ['ADADELTA']
all_costs = {}
all_accuracies = {}
for optimizer in optimizers:
    params_copy = params.copy()
    costs, accuracies = train(matrix_x_values, y_values, params_copy, epochs, learning_rate, optimizer)
    all_costs[optimizer] = costs
    all_accuracies[optimizer] = accuracies
```

    Epoch 0: Loss=0.17653394513381784, Accuracy=0.9141104294478528
    Epoch 100: Loss=0.16879310577636336, Accuracy=0.9263803680981595
    Epoch 200: Loss=0.16724377618147862, Accuracy=0.9263803680981595
    Epoch 300: Loss=0.165951313967353, Accuracy=0.9263803680981595
    Epoch 400: Loss=0.16465897862561643, Accuracy=0.9263803680981595
    Epoch 500: Loss=0.16365317756183798, Accuracy=0.9263803680981595
    Epoch 600: Loss=0.16277671497605545, Accuracy=0.9263803680981595
    Epoch 700: Loss=0.16156332334644716, Accuracy=0.9263803680981595
    Epoch 800: Loss=0.16044848109322685, Accuracy=0.9263803680981595
    Epoch 900: Loss=0.15932663270830294, Accuracy=0.9263803680981595
    Epoch 1000: Loss=0.15831221856929265, Accuracy=0.9263803680981595
    


```python
##################################
# Plotting the cost against iterations for
# AdaDelta Optimization
##################################
plt.figure(figsize=(10, 6))
for optimizer in optimizers:
    plt.plot(range(epochs), all_costs[optimizer], label=optimizer)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('ADADELTA Optimization: Cost Function by Iteration')
plt.ylim(0.15, 0.30)
plt.xlim(-50,1000)
plt.legend([], [], frameon=False)
plt.show()
```


    
![png](output_201_0.png)
    



```python
##################################
# Plotting the classification accuracy against iterations for
# AdaDelta Optimization
##################################
plt.figure(figsize=(10, 6))
for optimizer in optimizers:
    plt.plot(range(epochs), all_accuracies[optimizer], label=optimizer)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('ADADELTA Optimization: : Classification by Iteration')
plt.ylim(0.00, 1.00)
plt.xlim(-50,1000)
plt.legend([], [], frameon=False)
plt.show()
```


    
![png](output_202_0.png)
    



```python
##################################
# Gathering the final accuracy and cost values for 
# AdaDelta Optimization
##################################
ADADELTA_metrics = pd.DataFrame(["ACCURACY","LOSS"])
ADADELTA_values = pd.DataFrame([accuracies[-1],costs[-1]])
ADADELTA_method = pd.DataFrame(["ADADELTA"]*2)
ADADELTA_summary = pd.concat([ADADELTA_metrics, 
                         ADADELTA_values,
                         ADADELTA_method], axis=1)
ADADELTA_summary.columns = ['Metric', 'Value', 'Method']
ADADELTA_summary.reset_index(inplace=True, drop=True)
display(ADADELTA_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Value</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACCURACY</td>
      <td>0.926380</td>
      <td>ADADELTA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LOSS</td>
      <td>0.158312</td>
      <td>ADADELTA</td>
    </tr>
  </tbody>
</table>
</div>


### 1.6.6 Layer-wise Optimized Non-convex Optimization <a class="anchor" id="1.6.6"></a>

[Backpropagation](https://link.springer.com/book/10.1007/978-0-387-84858-7) and [Weight Update](https://link.springer.com/book/10.1007/978-0-387-84858-7), in the context of an artificial neural network, involve the process of iteratively adjusting the weights of the connections between neurons in the network to minimize the difference between the predicted and the actual target responses. Input data is fed into the neural network, and it propagates through the network layer by layer, starting from the input layer, through hidden layers, and ending at the output layer. At each neuron, the weighted sum of inputs is calculated, followed by the application of an activation function to produce the neuron's output. Once the forward pass is complete, the network's output is compared to the actual target output. The difference between the predicted output and the actual output is quantified using a loss function, which measures the discrepancy between the predicted and actual values. Common loss functions for classification tasks include cross-entropy loss. During the backward pass, the error is propagated backward through the network to compute the gradients of the loss function with respect to each weight in the network. This is achieved using the chain rule of calculus, which allows the error to be decomposed and distributed backward through the network. The gradients quantify how much a change in each weight would affect the overall error of the network. Once the gradients are computed, the weights are updated in the opposite direction of the gradient to minimize the error. This update is typically performed using an optimization algorithm such as gradient descent, which adjusts the weights in proportion to their gradients and a learning rate hyperparameter. The learning rate determines the size of the step taken in the direction opposite to the gradient. These steps are repeated for multiple iterations (epochs) over the training data. As the training progresses, the weights are adjusted iteratively to minimize the error, leading to a neural network model that accurately classifies input data.

[Optimization Algorithms](https://link.springer.com/book/10.1007/978-0-387-84858-7), in the context of neural network classification, are methods used to adjust the parameters (weights and biases) of a neural network during the training process in order to minimize a predefined loss function. The primary goal of these algorithms is to optimize the performance of the neural network by iteratively updating its parameters based on the feedback provided by the training data. Optimization algorithms play a critical role in the training of neural networks because they determine how effectively the network learns from the data and how quickly it converges to an optimal solution. These algorithms are significant during model development in improving model accuracy (optimization algorithms help improve the accuracy of neural network models by minimizing the classification error on the training data), enhancing generalization (by minimizing the loss function during training, optimization algorithms aim to generalize well to unseen data, thereby improving the model's ability to make accurate predictions on new inputs), reducing training time (efficient optimization algorithms can accelerate the convergence of the training process, leading to shorter training times for neural networks), handling complex data (since neural networks often deal with high-dimensional and non-linear data, optimization algorithms enable neural networks to effectively learn complex patterns and relationships within the data, leading to improved classification performance) and adapting to variations in data (optimization algorithms can adapt the model's parameters based on variations in the training data, ensuring robustness and stability in the face of different input distributions or data characteristics).

[Layer-wise Optimized Non-convex Optimization](https://link.springer.com/book/10.1007/978-0-387-84858-7) (Lion) focuses on adapting learning rates for each layer of the neural network based on the curvature of the loss landscape. Lion aims to accelerate convergence, improve optimization efficiency, and enhance the overall performance of deep neural networks in classification tasks.The Lion process involves adapting the learning rates for each layer of the neural network independently, allowing it to handle variations in the curvature and scale of the loss landscape across different layers, incorporating momentum-like updates to help accelerate convergence and navigate through the optimization space more efficiently and dynamically adjusting the learning rates based on the curvature of the loss landscape, ensuring that larger updates are made in regions with steep gradients and smaller updates in regions with shallow gradients. Lion demonstrates several advantages over other optimization methods in terms of layer-wise adaptation (Lion's ability to adapt learning rates layer-wise allows it to exploit the local curvature of the loss landscape, leading to more efficient optimization and faster convergence compared to methods with uniform learning rates.), efficient handling of deep architectures (Lion is specifically designed for training deep neural networks and can handle the challenges associated with deep architectures, such as vanishing gradients and optimization instabilities, more effectively than traditional optimization methods.), and enhanced generalization (Lion's adaptive learning rates and momentum-like updates help prevent overfitting and improve the generalization ability of the neural network classifier, leading to better performance on unseen data.). However, some disadvantages of Lion include complexity (Lion may have higher computational and implementation complexity compared to simpler optimization methods like SGD or AdaGrad. It requires careful tuning of hyperparameters and may be more challenging to implement correctly.), sensitivity to hyperparameters (Like many optimization algorithms, Lion's performance can be sensitive to the choice of hyperparameters, including the momentum parameter and the learning rate schedule. Finding the optimal hyperparameters may require extensive experimentation and tuning.), and limited practical evaluation ( Lion is a relatively new optimization algorithm, and its practical performance may not be extensively evaluated or well-understood compared to more established methods like SGD, Adam, or RMSprop.).

1. A neural network with the following structure was formulated:
    * <span style="color: #FF0000">Hidden Layer</span> = 3
    * <span style="color: #FF0000">Number of Nodes per Hidden Layer</span> = 5
2. The backpropagation and optimization algorithms were implemented with parameter settings described as follows:
    * <span style="color: #FF0000">Learning Rate</span> = 0.01
    * <span style="color: #FF0000">Epochs</span> = 1000
    * <span style="color: #FF0000">Hidden Layer Activation Function</span> = Sigmoid Activation Function
    * <span style="color: #FF0000">Output Layer Activation Function</span> = Softmax Activation Function
    * <span style="color: #FF0000">Loss Function Optimization Method</span> = Layer-wise Optimized Non-convex Optimization (LION)
    * <span style="color: #FF0000">Gamma (Exponential Decay of Accumulated Past Gradients)</span> = 0.999
    * <span style="color: #FF0000">Epsilon (Constant to Maintain Numerical Stability During Update)</span> = 1e-8
3. The final loss estimate determined as 0.15060 at the 1000th epoch was optimally low as compared to those obtained using the other optimization methods.
4. Applying parameter updates using an LION cost function optimization, the neural network model performance is estimated as follows:
    * <span style="color: #FF0000">Accuracy</span> = 93.25153
5. The estimated classification accuracy using the LION cost function optimization was optimal as compared to those obtained using the other optimization methods. 



```python
##################################
# Defining the function to implement
# Layer-wise Optimized Non-convex Optimization
##################################
def lion(params, gradients, learning_rate, z=None, r=None, gamma=0.999, eps=1e-8):
    if z is None:
        z = {key: np.zeros_like(value) for key, value in params.items()}
    if r is None:
        r = {key: np.zeros_like(value) for key, value in params.items()}
    
    for key in params.keys():
        z[key] = gamma * z[key] + (1 - gamma) * gradients['d' + key]
        r[key] = gamma * r[key] + (1 - gamma) * (gradients['d' + key] ** 2)
        delta = - learning_rate * z[key] / np.sqrt(r[key] + eps)
        params[key] += delta
```


```python
##################################
# Defining model training parameters
##################################
epochs = 1001
learning_rate = 0.01
```


```python
##################################
# Implementing the method on
# Layer-wise Optimized Non-convex Optimization
##################################
optimizers = ['LION']
all_costs = {}
all_accuracies = {}
for optimizer in optimizers:
    params_copy = params.copy()
    costs, accuracies = train(matrix_x_values, y_values, params_copy, epochs, learning_rate, optimizer)
    all_costs[optimizer] = costs
    all_accuracies[optimizer] = accuracies
```

    Epoch 0: Loss=0.15830151628894845, Accuracy=0.9263803680981595
    Epoch 100: Loss=0.15764517713374526, Accuracy=0.9263803680981595
    Epoch 200: Loss=0.15695744489638758, Accuracy=0.9263803680981595
    Epoch 300: Loss=0.15627166762892516, Accuracy=0.9263803680981595
    Epoch 400: Loss=0.15561303244310098, Accuracy=0.9263803680981595
    Epoch 500: Loss=0.15472009586203717, Accuracy=0.9263803680981595
    Epoch 600: Loss=0.15388359677979635, Accuracy=0.9263803680981595
    Epoch 700: Loss=0.15307060515557244, Accuracy=0.9263803680981595
    Epoch 800: Loss=0.152279697937234, Accuracy=0.9263803680981595
    Epoch 900: Loss=0.15145047879884138, Accuracy=0.9263803680981595
    Epoch 1000: Loss=0.15060131716040875, Accuracy=0.9325153374233128
    


```python
##################################
# Plotting the cost against iterations for
# Layer-wise Optimized Non-convex Optimization
##################################
plt.figure(figsize=(10, 6))
for optimizer in optimizers:
    plt.plot(range(epochs), all_costs[optimizer], label=optimizer)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('LION Optimization: Cost Function by Iteration')
plt.ylim(0.15, 0.30)
plt.xlim(-50,1000)
plt.legend([], [], frameon=False)
plt.show()
```


    
![png](output_208_0.png)
    



```python
##################################
# Plotting the classification accuracy against iterations for
# Layer-wise Optimized Non-convex Optimization
##################################
plt.figure(figsize=(10, 6))
for optimizer in optimizers:
    plt.plot(range(epochs), all_accuracies[optimizer], label=optimizer)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('LION Optimization: : Classification by Iteration')
plt.ylim(0.00, 1.00)
plt.xlim(-50,1000)
plt.legend([], [], frameon=False)
plt.show()
```


    
![png](output_209_0.png)
    



```python
##################################
# Gathering the final accuracy and cost values for 
# Layer-wise Optimized Non-convex Optimization
##################################
LION_metrics = pd.DataFrame(["ACCURACY","LOSS"])
LION_values = pd.DataFrame([accuracies[-1],costs[-1]])
LION_method = pd.DataFrame(["LION"]*2)
LION_summary = pd.concat([LION_metrics, 
                         LION_values,
                         LION_method], axis=1)
LION_summary.columns = ['Metric', 'Value', 'Method']
LION_summary.reset_index(inplace=True, drop=True)
display(LION_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Value</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACCURACY</td>
      <td>0.932515</td>
      <td>LION</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LOSS</td>
      <td>0.150601</td>
      <td>LION</td>
    </tr>
  </tbody>
</table>
</div>


### 1.6.7 Root Mean Square Propagation Optimization <a class="anchor" id="1.6.7"></a>

[Backpropagation](https://link.springer.com/book/10.1007/978-0-387-84858-7) and [Weight Update](https://link.springer.com/book/10.1007/978-0-387-84858-7), in the context of an artificial neural network, involve the process of iteratively adjusting the weights of the connections between neurons in the network to minimize the difference between the predicted and the actual target responses. Input data is fed into the neural network, and it propagates through the network layer by layer, starting from the input layer, through hidden layers, and ending at the output layer. At each neuron, the weighted sum of inputs is calculated, followed by the application of an activation function to produce the neuron's output. Once the forward pass is complete, the network's output is compared to the actual target output. The difference between the predicted output and the actual output is quantified using a loss function, which measures the discrepancy between the predicted and actual values. Common loss functions for classification tasks include cross-entropy loss. During the backward pass, the error is propagated backward through the network to compute the gradients of the loss function with respect to each weight in the network. This is achieved using the chain rule of calculus, which allows the error to be decomposed and distributed backward through the network. The gradients quantify how much a change in each weight would affect the overall error of the network. Once the gradients are computed, the weights are updated in the opposite direction of the gradient to minimize the error. This update is typically performed using an optimization algorithm such as gradient descent, which adjusts the weights in proportion to their gradients and a learning rate hyperparameter. The learning rate determines the size of the step taken in the direction opposite to the gradient. These steps are repeated for multiple iterations (epochs) over the training data. As the training progresses, the weights are adjusted iteratively to minimize the error, leading to a neural network model that accurately classifies input data.

[Optimization Algorithms](https://link.springer.com/book/10.1007/978-0-387-84858-7), in the context of neural network classification, are methods used to adjust the parameters (weights and biases) of a neural network during the training process in order to minimize a predefined loss function. The primary goal of these algorithms is to optimize the performance of the neural network by iteratively updating its parameters based on the feedback provided by the training data. Optimization algorithms play a critical role in the training of neural networks because they determine how effectively the network learns from the data and how quickly it converges to an optimal solution. These algorithms are significant during model development in improving model accuracy (optimization algorithms help improve the accuracy of neural network models by minimizing the classification error on the training data), enhancing generalization (by minimizing the loss function during training, optimization algorithms aim to generalize well to unseen data, thereby improving the model's ability to make accurate predictions on new inputs), reducing training time (efficient optimization algorithms can accelerate the convergence of the training process, leading to shorter training times for neural networks), handling complex data (since neural networks often deal with high-dimensional and non-linear data, optimization algorithms enable neural networks to effectively learn complex patterns and relationships within the data, leading to improved classification performance) and adapting to variations in data (optimization algorithms can adapt the model's parameters based on variations in the training data, ensuring robustness and stability in the face of different input distributions or data characteristics).

[Root Mean Square Propagation Optimization](https://link.springer.com/book/10.1007/978-0-387-84858-7) (RMSprop) addresses the limitations of AdaGrad, specifically the issue of diminishing learning rates over time, by introducing a decaying average of past squared gradients. RMSprop adjusts the learning rates for each parameter based on the root mean square of the gradients, allowing for more efficient optimization and faster convergence. The RMSprop process involves initializing the parameters, including a decaying average variable for the squared gradients (denoted as "cache") to zero, and setting a decay rate parameter, with value typically close to 1. In each iteration of training which is repeated until convergence or a predetermined number of iterations, the gradients of the loss function with respect to the parameters are computed, the decaying average of squared gradients is determined, and the parameters using the root mean square of the gradients are updated. RMSprop demonstrates several advantages over other optimization methods in terms of adaptive learning rates (RMSprop adapts learning rates for each parameter individually, making it less sensitive to manual tuning of learning rate hyperparameters compared to SGD.), efficient handling of noisy gradients (RMSprop performs well in scenarios with noisy gradients or non-stationary objectives. It adjusts the learning rates based on the root mean square of the gradients, effectively handling such gradients and improving optimization performance.), and prevention of diminishing learning rates (RMSprop prevents the issue of diminishing learning rates over time, which can occur in AdaGrad and RMSprop. This ensures that the learning rates remain relevant throughout the training process, leading to faster convergence and improved optimization efficiency.). However, some disadvantages of RMSprop include memory and computational cost (RMSprop requires additional memory and computation to store and update the decaying average of squared gradients for each parameter. This overhead can become significant for large-scale neural networks with a high number of parameters.), sensitivity to hyperparameters (the performance of RMSprop can be sensitive to the choice of hyperparameters, including the decay rate parameter. Finding the optimal hyperparameters may require extensive experimentation and tuning.), and potential overshooting (RMSprop's reliance on a decaying average of squared gradients may lead to overshooting or oscillations in the optimization process, especially in scenarios with highly non-convex objectives or noisy gradients.).

1. A neural network with the following structure was formulated:
    * <span style="color: #FF0000">Hidden Layer</span> = 3
    * <span style="color: #FF0000">Number of Nodes per Hidden Layer</span> = 5
2. The backpropagation and optimization algorithms were implemented with parameter settings described as follows:
    * <span style="color: #FF0000">Learning Rate</span> = 0.01
    * <span style="color: #FF0000">Epochs</span> = 1000
    * <span style="color: #FF0000">Hidden Layer Activation Function</span> = Sigmoid Activation Function
    * <span style="color: #FF0000">Output Layer Activation Function</span> = Softmax Activation Function
    * <span style="color: #FF0000">Loss Function Optimization Method</span> = Root Mean Square Propagation (RMSPROP)
    * <span style="color: #FF0000">Beta (Exponential Decay of the Average of Squared Gradients.)</span> = 0.900
    * <span style="color: #FF0000">Epsilon (Constant to Maintain Numerical Stability During Update)</span> = 1e-8
3. The final loss estimate determined as 0.18141 at the 1000th epoch was not optimally low as compared to those obtained using the other optimization methods.
4. Applying parameter updates using an RMSPROP cost function optimization, the neural network model performance is estimated as follows:
    * <span style="color: #FF0000">Accuracy</span> = 92.02453
5. The estimated classification accuracy using the RMSPROP cost function optimization was not optimal as compared to those obtained using the other optimization methods.



```python
##################################
# Defining the function to implement
# Root Mean Square Propagation Optimization
##################################
def rmsprop(params, gradients, learning_rate, cache=None, beta=0.9, eps=1e-8):
    if cache is None:
        cache = {k: np.zeros_like(v) for k, v in params.items()}

    for param_name in params:
        cache[param_name] = beta * cache[param_name] + (1 - beta) * (gradients['d' + param_name] ** 2)
        params[param_name] -= learning_rate * gradients['d' + param_name] / (np.sqrt(cache[param_name]) + eps)    
```


```python
##################################
# Defining model training parameters
##################################
epochs = 1001
learning_rate = 0.01
```


```python
##################################
# Implementing the method on
# Root Mean Square Propagation Optimization
##################################
optimizers = ['RMSPROP']
all_costs = {}
all_accuracies = {}
for optimizer in optimizers:
    params_copy = params.copy()
    costs, accuracies = train(matrix_x_values, y_values, params_copy, epochs, learning_rate, optimizer)
    all_costs[optimizer] = costs
    all_accuracies[optimizer] = accuracies
```

    Epoch 0: Loss=0.1505920847548153, Accuracy=0.9325153374233128
    Epoch 100: Loss=0.21546044742901144, Accuracy=0.8957055214723927
    Epoch 200: Loss=0.21991542558635138, Accuracy=0.8957055214723927
    Epoch 300: Loss=0.24190084144061785, Accuracy=0.8895705521472392
    Epoch 400: Loss=0.197519694701527, Accuracy=0.901840490797546
    Epoch 500: Loss=0.16744501199503276, Accuracy=0.9202453987730062
    Epoch 600: Loss=0.16407277026857114, Accuracy=0.9202453987730062
    Epoch 700: Loss=0.17830616454467615, Accuracy=0.9202453987730062
    Epoch 800: Loss=0.17992251695794698, Accuracy=0.9202453987730062
    Epoch 900: Loss=0.1808562606429272, Accuracy=0.9202453987730062
    Epoch 1000: Loss=0.18141333653994732, Accuracy=0.9202453987730062
    


```python
##################################
# Plotting the cost against iterations for
# Root Mean Square Propagation Optimization
##################################
plt.figure(figsize=(10, 6))
for optimizer in optimizers:
    plt.plot(range(epochs), all_costs[optimizer], label=optimizer)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('RMSPROP Optimization: Cost Function by Iteration')
plt.ylim(0.15, 0.30)
plt.xlim(-50,1000)
plt.legend([], [], frameon=False)
plt.show()
```


    
![png](output_215_0.png)
    



```python
##################################
# Plotting the classification accuracy against iterations for
# Root Mean Square Propagation Optimization
##################################
plt.figure(figsize=(10, 6))
for optimizer in optimizers:
    plt.plot(range(epochs), all_accuracies[optimizer], label=optimizer)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('RMSPROP Optimization: : Classification by Iteration')
plt.ylim(0.00, 1.00)
plt.xlim(-50,1000)
plt.legend([], [], frameon=False)
plt.show()
```


    
![png](output_216_0.png)
    



```python
##################################
# Gathering the final accuracy and cost values for 
# Root Mean Square Propagation Optimization
##################################

```


```python
RMSPROP_metrics = pd.DataFrame(["ACCURACY","LOSS"])
RMSPROP_values = pd.DataFrame([accuracies[-1],costs[-1]])
RMSPROP_method = pd.DataFrame(["RMSPROP"]*2)
RMSPROP_summary = pd.concat([RMSPROP_metrics, 
                         RMSPROP_values,
                         RMSPROP_method], axis=1)
RMSPROP_summary.columns = ['Metric', 'Value', 'Method']
RMSPROP_summary.reset_index(inplace=True, drop=True)
display(RMSPROP_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Value</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACCURACY</td>
      <td>0.920245</td>
      <td>RMSPROP</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LOSS</td>
      <td>0.181413</td>
      <td>RMSPROP</td>
    </tr>
  </tbody>
</table>
</div>


## 1.7. Consolidated Findings <a class="anchor" id="1.7"></a>

1. While all models showed comparably high classification accuracy, this optimization algorithm model demonstrated the lowest estimated cost values leading to the best discrimination between the dichotomous response.
    * <span style="color: #FF0000">LION</span> = Layer-wise Optimized Non-convex Optimization              
2. The choice of **Optimization Algorithm** can have a significant impact on the performance and training dynamics of a neural network classification model in terms of generalization ability, convergence speed, noise robustness, learning rate sensitivity, computational efficiency and training stability. The most appropriate algorithm should be carefully considered based on the specific characteristics of the dataset, model architecture, computational resources, and desired training objectives. Experimentation and empirical validation are often necessary to determine the most suitable optimization algorithm for a given neural network classification task.



```python
##################################
# Consolidating all the
# model performance metrics
##################################
model_performance_comparison = pd.concat([SGD_summary, 
                                          ADAM_summary,
                                          ADAGRAD_summary, 
                                          ADADELTA_summary,
                                          LION_summary, 
                                          RMSPROP_summary], 
                                         ignore_index=True)
print('Neural Network Model Comparison: ')
display(model_performance_comparison)
```

    Neural Network Model Comparison: 
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Value</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACCURACY</td>
      <td>0.926380</td>
      <td>SGD</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LOSS</td>
      <td>0.187134</td>
      <td>SGD</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ACCURACY</td>
      <td>0.914110</td>
      <td>ADAM</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LOSS</td>
      <td>0.176633</td>
      <td>ADAM</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACCURACY</td>
      <td>0.926380</td>
      <td>ADAGRAD</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LOSS</td>
      <td>0.172113</td>
      <td>ADAGRAD</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ACCURACY</td>
      <td>0.926380</td>
      <td>ADADELTA</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LOSS</td>
      <td>0.158312</td>
      <td>ADADELTA</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ACCURACY</td>
      <td>0.932515</td>
      <td>LION</td>
    </tr>
    <tr>
      <th>9</th>
      <td>LOSS</td>
      <td>0.150601</td>
      <td>LION</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ACCURACY</td>
      <td>0.920245</td>
      <td>RMSPROP</td>
    </tr>
    <tr>
      <th>11</th>
      <td>LOSS</td>
      <td>0.181413</td>
      <td>RMSPROP</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Consolidating the values for the
# accuracy metrics
# for all models
##################################
model_performance_comparison_accuracy = model_performance_comparison[model_performance_comparison['Metric']=='ACCURACY']
model_performance_comparison_accuracy.reset_index(inplace=True, drop=True)
model_performance_comparison_accuracy
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Value</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACCURACY</td>
      <td>0.926380</td>
      <td>SGD</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ACCURACY</td>
      <td>0.914110</td>
      <td>ADAM</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ACCURACY</td>
      <td>0.926380</td>
      <td>ADAGRAD</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACCURACY</td>
      <td>0.926380</td>
      <td>ADADELTA</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACCURACY</td>
      <td>0.932515</td>
      <td>LION</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ACCURACY</td>
      <td>0.920245</td>
      <td>RMSPROP</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Plotting the values for the
# accuracy metrics
# for all models
##################################
fig, ax = plt.subplots(figsize=(7, 7))
accuracy_hbar = ax.barh(model_performance_comparison_accuracy['Method'], model_performance_comparison_accuracy['Value'])
ax.set_xlabel("Accuracy")
ax.set_ylabel("Neural Network Classification Models")
ax.bar_label(accuracy_hbar, fmt='%.5f', padding=-50, color='white', fontweight='bold')
ax.set_xlim(0,1)
plt.show()
```


    
![png](output_222_0.png)
    



```python
##################################
# Consolidating the values for the
# logarithmic loss error metrics
# for all models
##################################
model_performance_comparison_loss = model_performance_comparison[model_performance_comparison['Metric']=='LOSS']
model_performance_comparison_loss.reset_index(inplace=True, drop=True)
model_performance_comparison_loss
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Value</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LOSS</td>
      <td>0.187134</td>
      <td>SGD</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LOSS</td>
      <td>0.176633</td>
      <td>ADAM</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LOSS</td>
      <td>0.172113</td>
      <td>ADAGRAD</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LOSS</td>
      <td>0.158312</td>
      <td>ADADELTA</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LOSS</td>
      <td>0.150601</td>
      <td>LION</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LOSS</td>
      <td>0.181413</td>
      <td>RMSPROP</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Plotting the values for the
# loss error
# for all models
##################################
fig, ax = plt.subplots(figsize=(7, 7))
loss_hbar = ax.barh(model_performance_comparison_loss['Method'], model_performance_comparison_loss['Value'])
ax.set_xlabel("Loss Error")
ax.set_ylabel("Neural Network Classification Models")
ax.bar_label(loss_hbar, fmt='%.5f', padding=-50, color='white', fontweight='bold')
ax.set_xlim(0,0.20)
plt.show()
```


    
![png](output_224_0.png)
    


# 2. Summary <a class="anchor" id="Summary"></a>


![Project49_Summary.png](attachment:ac057eaa-b240-4d7e-a391-52ed03b3ae5e.png)

# 3. References <a class="anchor" id="References"></a>

* **[Book]** [Deep Learning: A Visual Approach](https://nostarch.com/deep-learning-visual-approach) by Andrew Glassner
* **[Book]** [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python-second-edition) by François Chollet
* **[Book]** [The Elements of Statistical Learning](https://link.springer.com/book/10.1007/978-0-387-84858-7) by Trevor Hastie, Robert Tibshirani and Jerome Friedman
* **[Book]** [Data Preparation for Machine Learning: Data Cleaning, Feature Selection, and Data Transforms in Python](https://machinelearningmastery.com/data-preparation-for-machine-learning/) by Jason Brownlee
* **[Book]** [Feature Engineering and Selection: A Practical Approach for Predictive Models](http://www.feat.engineering/) by Max Kuhn and Kjell Johnson
* **[Book]** [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) by Alice Zheng and Amanda Casari
* **[Book]** [Applied Predictive Modeling](https://link.springer.com/book/10.1007/978-1-4614-6849-3?page=1) by Max Kuhn and Kjell Johnson
* **[Book]** [Data Mining: Practical Machine Learning Tools and Techniques](https://www.sciencedirect.com/book/9780123748560/data-mining-practical-machine-learning-tools-and-techniques?via=ihub=) by Ian Witten, Eibe Frank, Mark Hall and Christopher Pal 
* **[Book]** [Data Cleaning](https://dl.acm.org/doi/book/10.1145/3310205) by Ihab Ilyas and Xu Chu
* **[Book]** [Data Wrangling with Python](https://www.oreilly.com/library/view/data-wrangling-with/9781491948804/) by Jacqueline Kazil and Katharine Jarmul
* **[Book]** [Regression Modeling Strategies](https://link.springer.com/book/10.1007/978-1-4757-3462-1) by Frank Harrell
* **[Python Library API]** [NumPy](https://numpy.org/doc/) by NumPy Team
* **[Python Library API]** [pandas](https://pandas.pydata.org/docs/) by Pandas Team
* **[Python Library API]** [seaborn](https://seaborn.pydata.org/) by Seaborn Team
* **[Python Library API]** [matplotlib.pyplot](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html) by MatPlotLib Team
* **[Python Library API]** [itertools](https://docs.python.org/3/library/itertools.html) by Python Team
* **[Python Library API]** [operator](https://docs.python.org/3/library/operator.html) by Python Team
* **[Python Library API]** [sklearn.experimental](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.experimental) by Scikit-Learn Team
* **[Python Library API]** [sklearn.impute](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.impute) by Scikit-Learn Team
* **[Python Library API]** [sklearn.linear_model](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model) by Scikit-Learn Team
* **[Python Library API]** [sklearn.preprocessing](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) by Scikit-Learn Team
* **[Python Library API]** [scipy](https://docs.scipy.org/doc/scipy/) by SciPy Team
* **[Article]** [Step-by-Step Exploratory Data Analysis (EDA) using Python](https://www.analyticsvidhya.com/blog/2022/07/step-by-step-exploratory-data-analysis-eda-using-python/#:~:text=Exploratory%20Data%20Analysis%20(EDA)%20with,distributions%20using%20Python%20programming%20language.) by Malamahadevan Mahadevan (Analytics Vidhya)
* **[Article]** [Exploratory Data Analysis in Python — A Step-by-Step Process](https://towardsdatascience.com/exploratory-data-analysis-in-python-a-step-by-step-process-d0dfa6bf94ee) by Andrea D'Agostino (Towards Data Science)
* **[Article]** [Exploratory Data Analysis with Python](https://medium.com/@douglas.rochedo/exploratory-data-analysis-with-python-78b6c1d479cc) by Douglas Rocha (Medium)
* **[Article]** [4 Ways to Automate Exploratory Data Analysis (EDA) in Python](https://builtin.com/data-science/EDA-python) by Abdishakur Hassan (BuiltIn)
* **[Article]** [10 Things To Do When Conducting Your Exploratory Data Analysis (EDA)](https://www.analyticsvidhya.com) by Alifia Harmadi (Medium)
* **[Article]** [How to Handle Missing Data with Python](https://machinelearningmastery.com/handle-missing-data-python/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Statistical Imputation for Missing Values in Machine Learning](https://machinelearningmastery.com/statistical-imputation-for-missing-values-in-machine-learning/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Imputing Missing Data with Simple and Advanced Techniques](https://towardsdatascience.com/imputing-missing-data-with-simple-and-advanced-techniques-f5c7b157fb87) by Idil Ismiguzel (Towards Data Science)
* **[Article]** [Missing Data Imputation Approaches | How to handle missing values in Python](https://www.machinelearningplus.com/machine-learning/missing-data-imputation-how-to-handle-missing-values-in-python/) by Selva Prabhakaran (Machine Learning +)
* **[Article]** [Master The Skills Of Missing Data Imputation Techniques In Python(2022) And Be Successful](https://medium.com/analytics-vidhya/a-quick-guide-on-missing-data-imputation-techniques-in-python-2020-5410f3df1c1e) by Mrinal Walia (Analytics Vidhya)
* **[Article]** [How to Preprocess Data in Python](https://builtin.com/machine-learning/how-to-preprocess-data-python) by Afroz Chakure (BuiltIn)
* **[Article]** [Easy Guide To Data Preprocessing In Python](https://www.kdnuggets.com/2020/07/easy-guide-data-preprocessing-python.html) by Ahmad Anis (KDNuggets)
* **[Article]** [Data Preprocessing in Python](https://towardsdatascience.com/data-preprocessing-in-python-b52b652e37d5) by Tarun Gupta (Towards Data Science)
* **[Article]** [Data Preprocessing using Python](https://medium.com/@suneet.bhopal/data-preprocessing-using-python-1bfee9268fb3) by Suneet Jain (Medium)
* **[Article]** [Data Preprocessing in Python](https://medium.com/@abonia/data-preprocessing-in-python-1f90d95d44f4) by Abonia Sojasingarayar (Medium)
* **[Article]** [Data Preprocessing in Python](https://medium.datadriveninvestor.com/data-preprocessing-3cd01eefd438) by Afroz Chakure (Medium)
* **[Article]** [Detecting and Treating Outliers | Treating the Odd One Out!](https://www.analyticsvidhya.com/blog/2021/05/detecting-and-treating-outliers-treating-the-odd-one-out/) by Harika Bonthu (Analytics Vidhya)
* **[Article]** [Outlier Treatment with Python](https://medium.com/analytics-vidhya/outlier-treatment-9bbe87384d02) by Sangita Yemulwar (Analytics Vidhya)
* **[Article]** [A Guide to Outlier Detection in Python](https://builtin.com/data-science/outlier-detection-python) by Sadrach Pierre (BuiltIn)
* **[Article]** [How To Find Outliers in Data Using Python (and How To Handle Them)](https://careerfoundry.com/en/blog/data-analytics/how-to-find-outliers/) by Eric Kleppen (Career Foundry)
* **[Article]** [Statistics in Python — Collinearity and Multicollinearity](https://towardsdatascience.com/statistics-in-python-collinearity-and-multicollinearity-4cc4dcd82b3f) by Wei-Meng Lee (Towards Data Science)
* **[Article]** [Understanding Multicollinearity and How to Detect it in Python](https://towardsdatascience.com/everything-you-need-to-know-about-multicollinearity-2f21f082d6dc) by Terence Shin (Towards Data Science)
* **[Article]** [A Python Library to Remove Collinearity](https://www.yourdatateacher.com/2021/06/28/a-python-library-to-remove-collinearity/) by Gianluca Malato (Your Data Teacher)
* **[Article]** [8 Best Data Transformation in Pandas](https://ai.plainenglish.io/data-transformation-in-pandas-29b2b3c61b34) by Tirendaz AI (Medium)
* **[Article]** [Data Transformation Techniques with Python: Elevate Your Data Game!](https://medium.com/@siddharthverma.er.cse/data-transformation-techniques-with-python-elevate-your-data-game-21fcc7442cc2) by Siddharth Verma (Medium)
* **[Article]** [Data Scaling with Python](https://www.kdnuggets.com/2023/07/data-scaling-python.html) by Benjamin Obi Tayo (KDNuggets)
* **[Article]** [How to Use StandardScaler and MinMaxScaler Transforms in Python](https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Feature Engineering: Scaling, Normalization, and Standardization](https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/) by Aniruddha Bhandari  (Analytics Vidhya)
* **[Article]** [How to Normalize Data Using scikit-learn in Python](https://www.digitalocean.com/community/tutorials/normalize-data-in-python) by Jayant Verma (Digital Ocean)
* **[Article]** [What are Categorical Data Encoding Methods | Binary Encoding](https://www.analyticsvidhya.com/blog/2020/08/types-of-categorical-data-encoding/) by Shipra Saxena  (Analytics Vidhya)
* **[Article]** [Guide to Encoding Categorical Values in Python](https://pbpython.com/categorical-encoding.html) by Chris Moffitt (Practical Business Python)
* **[Article]** [Categorical Data Encoding Techniques in Python: A Complete Guide](https://soumenatta.medium.com/categorical-data-encoding-techniques-in-python-a-complete-guide-a913aae19a22) by Soumen Atta (Medium)
* **[Article]** [Categorical Feature Encoding Techniques](https://towardsdatascience.com/categorical-encoding-techniques-93ebd18e1f24) by Tara Boyle (Medium)
* **[Article]** [Ordinal and One-Hot Encodings for Categorical Data](https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Hypothesis Testing with Python: Step by Step Hands-On Tutorial with Practical Examples](https://towardsdatascience.com/hypothesis-testing-with-python-step-by-step-hands-on-tutorial-with-practical-examples-e805975ea96e) by Ece Işık Polat (Towards Data Science)
* **[Article]** [17 Statistical Hypothesis Tests in Python (Cheat Sheet)](https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [A Step-by-Step Guide to Hypothesis Testing in Python using Scipy](https://medium.com/@gabriel_renno/a-step-by-step-guide-to-hypothesis-testing-in-python-using-scipy-8eb5b696ab07) by Gabriel Rennó (Medium)
* **[Article]** [How Does Backpropagation in a Neural Network Work?](https://builtin.com/machine-learning/backpropagation-neural-network) by Anas Al-Masri (Builtin)
* **[Article]** [A Step by Step Backpropagation Example](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/) by Matt Mazur (MattMazur.Com)
* **[Article]** [Understanding Backpropagation](https://towardsdatascience.com/understanding-backpropagation-abcc509ca9d0) by Brent Scarff (Towards Data Science)
* **[Article]** [Understanding Backpropagation Algorithm](https://towardsdatascience.com/understanding-backpropagation-algorithm-7bb3aa2f95fd) by Simeon Kostadinov (Towards Data Science)
* **[Article]** [A Comprehensive Guide to the Backpropagation Algorithm in Neural Networks](https://neptune.ai/blog/backpropagation-algorithm-in-neural-networks-guide) by Ahmed Gad (Neptune.AI)
* **[Article]** [Backpropagation](https://brilliant.org/wiki/backpropagation/) by John McGonagle, George Shaikouski and Christopher Williams (Brilliant)
* **[Article]** [Backpropagation in Neural Networks](https://serokell.io/blog/understanding-backpropagation) by Inna Logunova (Serokell.IO)
* **[Article]** [Backpropagation Concept Explained in 5 Levels of Difficulty](https://medium.com/coinmonks/backpropagation-concept-explained-in-5-levels-of-difficulty-8b220a939db5) by Devashish Sood (Medium)
* **[Article]** [BackProp Explainer](https://xnought.github.io/backprop-explainer/) by Donny Bertucci (GitHub)
* **[Article]** [Backpropagation Algorithm in Neural Network and Machine Learning](https://intellipaat.com/blog/tutorial/artificial-intelligence-tutorial/back-propagation-algorithm/) by Intellipaat Team
* **[Article]** [Understanding Backpropagation in Neural Networks](https://ai.plainenglish.io/understanding-backpropagation-in-neural-networks-3634aad3a3c4) by Tech-AI-Math Team
* **[Article]** [Backpropagation Neural Network using Python](https://medium.com/@gabriel_renno/a-step-by-step-guide-to-hypothesis-testing-in-python-using-scipy-8eb5b696ab07) by Avinash Navlani (Machine Learning Geek)
* **[Article]** [Back Propagation in Neural Network: Machine Learning Algorithm](https://www.guru99.com/backpropogation-neural-network.html) by Daniel Johnson (Guru99)
* **[Article]** [What is Backpropagation?](https://deepai.org/machine-learning-glossary-and-terms/backpropagation) by Thomas Wood (DeepAI.Org)
* **[Article]** [Activation Functions in Neural Networks [12 Types & Use Cases]](https://www.v7labs.com/blog/neural-networks-activation-functions) by Pragati Baheti (V7.Com)
* **[Article]** [Activation Functions in Neural Networks](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6) by Sagar Sharma (Towards Data Science)
* **[Article]** [Comparison of Sigmoid, Tanh and ReLU Activation Functions](https://www.aitude.com/comparison-of-sigmoid-tanh-and-relu-activation-functions/) by Sandeep Kumar (AItude.Com)
* **[Article]** [How to Choose an Activation Function for Deep Learning](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Choosing the Right Activation Function in Deep Learning: A Practical Overview and Comparison](https://python.plainenglish.io/choosing-the-right-activation-function-in-deep-learning-a-practical-overview-and-comparison-c5372fe167f8) by Okan Yenigün (Medium)
* **[Article]** [Activation Functions in Neural Networks](https://www.geeksforgeeks.org/activation-functions-neural-networks/) by Geeks For Geeks Team
* **[Article]** [A Practical Comparison of Activation Functions](https://becominghuman.ai/a-practical-comparison-of-activation-functions-6f348be7d274) by Danny Denenberg (Medium)
* **[Article]** [Activation Functions in Neural Networks: With 15 examples](https://encord.com/blog/activation-functions-neural-networks/) by Nikolaj Buhl (Encord.Com)
* **[Article]** [Activation functions used in Neural Networks - Which is Better?](https://anishsinghwalia.medium.com/activation-functions-used-in-neural-networks-which-is-better-d4817d91199a) by Anish Singh Walia (Medium)
* **[Article]** [6 Types of Activation Function in Neural Networks You Need to Know](https://www.upgrad.com/blog/types-of-activation-function-in-neural-networks/) by Kechit Goyal (UpGrad.Com)
* **[Article]** [Activation Functions in Neural Networks](https://www.superannotate.com/blog/activation-functions-in-neural-networks) by SuperAnnotate Team
* **[Article]** [Compare Activation Layers](https://www.mathworks.com/help/deeplearning/ug/compare-activation-layers.html) by MathWorks Team
* **[Article]** [Activation Functions In Neural Networks](https://www.comet.com/site/blog/activation-functions-in-neural-networks/) by Kurtis Pykes (Comet.Com)
* **[Article]** [ReLU vs. Sigmoid Function in Deep Neural Networks](https://wandb.ai/ayush-thakur/dl-question-bank/reports/ReLU-vs-Sigmoid-Function-in-Deep-Neural-Networks--VmlldzoyMDk0MzI) by Ayush Thakur (Wanb.AI)
* **[Article]** [Using Activation Functions in Neural Networks](https://machinelearningmastery.com/using-activation-functions-in-neural-networks/) by Jason Bronwlee (Machine Learning Mastery)
* **[Article]** [Activation Function: Top 9 Most Popular Explained & When To Use Them](https://spotintelligence.com/2023/06/16/activation-function/) by Neri Van Otten (SpotIntelligence.Com)
* **[Article]** [5 Deep Learning and Neural Network Activation Functions to Know](https://builtin.com/machine-learning/activation-functions-deep-learning) by Artem Oppermann (BuiltIn.Com)
* **[Article]** [Activation Functions in Deep Learning: Sigmoid, tanh, ReLU](https://artemoppermann.com/activation-functions-in-deep-learning-sigmoid-tanh-relu/) by Artem Oppermann
* **[Article]** [7 Types of Activation Functions in Neural Network](https://www.analyticssteps.com/blogs/7-types-activation-functions-neural-network) by Dinesh Kumawat (AnalyticsSteps.Com)
* **[Article]** [What is an Activation Function? A Complete Guide](https://blog.roboflow.com/activation-function-computer-vision/#:~:text=Petru%20Potrimba,of%20deep%20learning%20model%20outputs.) by Petru Potrimba (RoboFlow.Com)
* **[Article]** [Various Optimization Algorithms For Training Neural Network](https://towardsdatascience.com/optimizers-for-training-neural-network-59450d71caf6) by Sanket Doshi (Towards Data Science)
* **[Article]** [Optimization Algorithms in Neural Networks](https://www.kdnuggets.com/2020/12/optimization-algorithms-neural-networks.html) by Nagesh Singh Chauhan (KDNuggets)
* **[Article]** [A Comprehensive Guide on Optimizers in Deep Learning](https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-deep-learning-optimizers/) by Ayush Gupta (Analytics Vidhya)
* **[Article]** [How to Manually Optimize Neural Network Models](https://machinelearningmastery.com/manually-optimize-neural-networks/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [How to Choose an Optimization Algorithm](https://machinelearningmastery.com/tour-of-optimization-algorithms/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Types of Optimization Algorithms used in Neural Networks and Ways to Optimize Gradient Descent](https://medium.com/nerd-for-tech/types-of-optimization-algorithms-used-in-neural-networks-and-ways-to-optimize-gradient-descent-1e32cdcbcf6c) by Anish Singh Walia (Medium)
* **[Article]** [Optimizing Neural Networks: Strategies and Techniques](https://ai.plainenglish.io/optimizing-neural-networks-strategies-and-techniques-7097f85898f0) by Kajeeth Kumar (AI.PlainEnglish.IO)
* **[Article]** [Neural Network Optimization](https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0) by Matthew Stewart (Towards Data Science)
* **[Article]** [Optimizers in Deep Learning](https://www.scaler.com/topics/deep-learning/optimizers-in-deep-learning/) by Cathrine Jeeva (Scaler)
* **[Article]** [Understanding Deep Learning Optimizers: Momentum, AdaGrad, RMSProp & Adam](https://towardsdatascience.com/understanding-deep-learning-optimizers-momentum-adagrad-rmsprop-adam-e311e377e9c2) by Vyacheslav Efimov (Towards Data Science)
* **[Article]** [Types of Optimizers in Deep Learning Every AI Engineer Should Know](https://www.upgrad.com/blog/types-of-optimizers-in-deep-learning/) by Pavan Vadapalli (UpGrad.Com)
* **[Article]** [Optimizers](https://ml-cheatsheet.readthedocs.io/en/latest/optimizers.html) by ML-Cheatsheet Team
* **[Article]** [Optimizers](https://docs.edgeimpulse.com/docs/concepts/ml-concepts/neural-networks/optimizers) by Edge Info Team
* **[Article]** [Navigating Neural Network Optimization: A Comprehensive Guide to Types of Optimizers](https://medium.com/@l228146/navigating-neural-network-optimization-a-comprehensive-guide-to-types-of-optimizers-404042b03940) by Muhammad Zain Tariq (Medium)
* **[Article]** [An Introduction to Artificial Neural Network Optimizers](https://www.pradeep-natarajan.com/blog/Neural-Network-Optimizers) by Pradeep Natarajan
* **[Article]** [Optimizers Explained for Training Neural Networks](https://dropsofai.com/optimizers-explained-for-training-neural-networks/#google_vignette) by Kartik Chaudhary (DropsOfAI.Com)
* **[Article]** [Optimizers in Neural Networks](https://blog.aiensured.com/optimizers-in-neural-networks/) by AI Ensured Team
* **[Article]** [Parameter Optimization in Neural Networks](https://www.deeplearning.ai/ai-notes/optimization/index.html) by DeepLearning.AI Team
* **[Article]** [Overview of Different Optimizers for Neural Networks](https://medium.datadriveninvestor.com/overview-of-different-optimizers-for-neural-networks-e0ed119440c3) by Renu Khandelwal (Medium)
* **[Article]** [Which Optimizer should I use for my ML Project?](https://www.lightly.ai/post/which-optimizer-should-i-use-for-my-machine-learning-project) by Lightly.AI Team
* **[Article]** [A Journey into Optimization Algorithms for Deep Neural Networks](https://theaisummer.com/optimization/) by Sergios Karagiannakos (AISummer.Com)
* **[Article]** [How to Compare Keras Optimizers in Tensorflow for Deep Learning](https://wandb.ai/sauravm/Optimizers/reports/How-to-Compare-Keras-Optimizers-in-Tensorflow-for-Deep-Learning--VmlldzoxNjU1OTA4) by Saurav Maheshkar (WAndB.Com)
* **[Article]** [An Empirical Comparison of Optimizers for Machine Learning Models](https://heartbeat.comet.ml/an-empirical-comparison-of-optimizers-for-machine-learning-models-b86f29957050) by Rick Wierenga (Medium)
* **[Publication]** [Data Quality for Machine Learning Tasks](https://journals.sagepub.com/doi/10.1177/0962280206074463) by Nitin Gupta, Shashank Mujumdar, Hima Patel, Satoshi Masuda, Naveen Panwar, Sambaran Bandyopadhyay, Sameep Mehta, Shanmukha Guttula, Shazia Afzal, Ruhi Sharma Mittal and Vitobha Munigala (KDD ’21: Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining)
* **[Publication]** [Overview and Importance of Data Quality for Machine Learning Tasks](https://dl.acm.org/doi/10.1145/3394486.3406477) by Abhinav Jain, Hima Patel, Lokesh Nagalapatti, Nitin Gupta, Sameep Mehta, Shanmukha Guttula, Shashank Mujumdar, Shazia Afzal, Ruhi Sharma Mittal and Vitobha Munigala (KDD ’20: Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining)
* **[Publication]** [Multiple Imputation of Discrete and Continuous Data by Fully Conditional Specification](https://journals.sagepub.com/doi/10.1177/0962280206074463) by Stef van Buuren (Statistical Methods in Medical Research)
* **[Publication]** [Mathematical Contributions to the Theory of Evolution: Regression, Heredity and Panmixia](https://royalsocietypublishing.org/doi/10.1098/rsta.1896.0007) by Karl Pearson (Royal Society)
* **[Publication]** [A New Family of Power Transformations to Improve Normality or Symmetry](https://academic.oup.com/biomet/article-abstract/87/4/954/232908?redirectedFrom=fulltext&login=false) by In-Kwon Yeo and Richard Johnson (Biometrika)
* **[Course]** [IBM Data Analyst Professional Certificate](https://www.coursera.org/professional-certificates/ibm-data-analyst) by IBM Team (Coursera)
* **[Course]** [IBM Data Science Professional Certificate](https://www.coursera.org/professional-certificates/ibm-data-science) by IBM Team (Coursera)
* **[Course]** [IBM Machine Learning Professional Certificate](https://www.coursera.org/professional-certificates/ibm-machine-learning) by IBM Team (Coursera)
* **[Course]** [Machine Learning Specialization Certificate](https://www.coursera.org/specializations/machine-learning-introduction) by DeepLearning.AI Team (Coursera)

***


```python
from IPython.display import display, HTML
display(HTML("<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>"))
```


<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>

