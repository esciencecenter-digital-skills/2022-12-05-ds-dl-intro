![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document Day 1

2022-12-05-ds-dl-intro

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: [link](https://tinyurl.com/2022-12-05-ds-dl-intro-day1)

Collaborative Document day 1: [link](https://tinyurl.com/2022-12-05-ds-dl-intro-day1)

Collaborative Document day 2: [link](https://tinyurl.com/2022-12-05-ds-dl-intro-day2)

Collaborative Document day 3: [link](https://tinyurl.com/2022-12-05-ds-dl-intro-day3)

Collaborative Document day 4: [link](https://tinyurl.com/2022-12-05-ds-dl-intro-day4) 

## 👮Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.
 
## 🎓 Certificate of attendance

If you attend the full workshop you can request a certificate of attendance by emailing to training@esciencecenter.nl .

## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question, raise your hand in zoom. Click on the icon labeled "Reactions" in the toolbar on the bottom center of your screen,
then click the button 'Raise Hand ✋'. For urgent questions, just unmute and speak up!

You can also ask questions or type 'I need help' in the chat window and helpers will try to help you.
Please note it is not necessary to monitor the chat - the helpers will make sure that relevant questions are addressed in a plenary way.
(By the way, off-topic questions will still be answered in the chat)


## 🖥 Workshop website

[link](https://esciencecenter-digital-skills.github.io/2022-12-05-ds-dl-intro/)

🛠 Setup

[link](https://esciencecenter-digital-skills.github.io/2022-12-05-ds-dl-intro/#setup)

Download files

[link](https://esciencecenter-digital-skills.github.io/2022-12-05-ds-dl-intro/#setup)

## 👩‍🏫👩‍💻🎓 Instructors

Djura Smits, Cunliang Geng, Pranav Chandramouli

## 🧑‍🙋 Helpers

Robin Richardson, Ole Mussmann


## 🗓️ Agenda
| Time | Topic |
|--:|:---|
| 9:00 | 	Welcome and icebreaker |
| 9:15 | Introduction  |
| 10:15 | Coffee break |
| 10:30 | Classification by a Neural Network using Keras|
|11:30 | Tea break |
|11:45 | Classification, continued |
|12:45 | Wrapping up|
|13:00 | End |

## 🔧 Exercises
### Calculate the output for one neuron
Suppose we have

Input: X = (0, 0.5, 1)
Weights: W = (-1, -0.5, 0.5)
Bias: b = 1
Activation function relu: f(x) = max(x, 0)
What is the output of the neuron?

*Note: You can use whatever you like: brain only, pen&paper, Python, Excel…*

| Name | Answer |
| ---- | ----- |
|Anita| 1.25|
|Ahnjili| 1.25|
|Bram|1.25 |
|Erik| 1.5|
|Fleur|1.25 |
|Joshua|1.25 |
|Néstor|1.25 |
|Ralph| 1.5 |
|Sophie| 1.25 |
|Tadzio|1.25 |
|Tom| 1.25|
|Venustiano| 1.25 |
|Saba| 1.25|
|Wenjie| 1.5|


## 🧠 Collaborative Notes

### :wave: Introduction
- Deep learning is a sub-category of machine learning
- Use multi-layer neural networks to learn from vast amount of data
- Design is inspired by the human brain

#### :spider_web: Neural Networks (NN)
- Neural Networks have layers:
  - One or more inputs (X), scaled by weights (W) (floating point numbers)
  - Add them together with a fixed bias (b)
    - output = W1\*X1 + W2\*X2 + … + b\*W(n+1) 
  - Send the sum through a non-linear "activation function", producing one or more output values
- NN can have many connected layers
- More layers take longer to train and do not necessarily perform better, a good balance is important

#### :bulb: What Problems Can Deep Learning Solve?
- Pattern/object recognition
- Segmenting images or data
- Translating between one set of data and another, e.g. language translation
- Generating new data that looks similar to training data
  - Create synthetic datasets
  - Increase resolution of existing data
  - "Deep fake" images/videos
- Problems that are not sharply defined
  - But you still need labelled data to confirm it's working

#### :confused: Where Does Deep Learning Struggle?
- Cases with little training data
- Tasks requiring an explanation of how the answer was arrived at
  - It's to some extent a "closed box"
  - There are projects working on "explainable AI", see [DIANNA](https://github.com/dianna-ai/dianna)
- Classifying things which differ from the training data

#### :collision: When is Deep Learning Overkill?
- Logic operations, like
  - computing totals, averages, ranges, etc.
- Modelling well-defined systems
- Basic computer vision tasks, like
  - edge detection
  - decreasing color depth
  - blurring an image

#### 🌊 Deep Learning Workflow (Ten Steps to Success)
- Formulate the problem
  - Is it regression? Classification? Something else?
- Identify inputs and outputs
  - What features can we use as input?
  - What do we want to get out of the network?
- Prepare data
  - Neural networks need numerical data
  - Split dataset training/test
- Choose a pre-trained model
  - ... or build a new architecture from scratch
- Choose a loss function and optimizer
  - Might need some experimentation
- Train model
  - Monitor progress to see if you are on the right track
  - How close to the "truth" (labels) is my model?
- Perform a prediction / classification
- Measure performance
- Tune hyperparameters
  - Other aspects and settings, beyond weights and biases
  - Number of layers, number of nodes per layers, number of epochs, ...
  - Loss function
  - Optimizer
- Share the model
  - So other's can benefit from your hard work
  - Include architecture, weights

#### :open_book: Deep Learning Libraries
- [TensorFlow](https://www.tensorflow.org/)
  - Rather low-level
  - Good integration with accelerators, like GPUs
- [PyTorch](https://pytorch.org/)
  - Developed for Python
  - Higher level variant: [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [Keras](https://keras.io/)
  - Easy to get started
  - Slower performance for huge amounts of data / large networks
- ([scikit-learn](https://scikit-learn.org/stable/index.html))
  - machine learning, but not deep learning

#### :alembic: Test Your Deep Learning Environment
- Activate your `conda` environment, if necessary
- Start Jupyter: `jupyter lab`
- Create new notebook
- Type into a cell and execute the following code blocks:
```
import sklearn
print('sklearn version: ', sklearn.__version__)
```
```
import seaborn
print('seaborn version: ', seaborn.__version__)
```
```
import pandas
print('pandas version: ', pandas.__version__)
```
```
from tensorflow import keras
print('Keras version: ', keras.__version__)
```
```
import tensorflow
print('Tensorflow version: ', tensorflow.__version__) 
```

### :nerd_face: Classification by a Neural Network using Keras
#### :spiral_note_pad: Formulate the Problem
> Goal: Predict species of penguins from a public dataset

![penguins](https://i.imgur.com/BoCLwxW.png)

![penguins2](https://carpentries-incubator.github.io/deep-learning-intro/fig/culmen_depth.png)
#### :mag_right: Identify Inputs and Outputs
```
import seaborn as sns

penguins = sns.load_dataset('penguins')
type(penguins)
```
- In Jupyter, get an overview of the data with
```
penguins
```
- Look out for missing values or errors in the dataset
- Get statistics of numerical features
```
penguins.describe()
```
- What do we want to predict?
```
penguins['species'].unique()
```
- Get a graphical overview of the data
```
sns.pairplot(penguins, hue="species")
```

### :cooking: Prepare the Data
- If needed, change data types to categories
```
penguins['species'] = penguins['species'].astype('category')
```
- Check the type again with
```
penguins['species']
```
#### Filter Data, Clean Missing Values
```
penguins_filtered = penguins.drop(columns=["island", "sex"]).dropna()
```
- Have a look at the filtered dataset
```
penguins_filtered
```
- Remove the species column we want to predict
```
penguins_features = penguins_filtered.drop(columns=["species"])
```
- Check the first few lines with
```
penguins_features.head()
```
#### Prepare Target Data
- We cannot predict the species _strings_, so we have to convert it to numerical
- For this we use [one hot encoding](https://www.geeksforgeeks.org/ml-one-hot-encoding-of-datasets-in-python/)
```
import pandas as pd
target = pd.get_dummies(penguins_filtered["species"])
```
- Check out what we just created and look at the last few lines
```
target.tail()
```
```
penguins_filtered["species"].tail()
```
#### Prepare Training and Test Set
```
from sklearn.model_selection import train_test_split
```
- Our test set will be 20% (0.2) of our whole dataset
- We set a `random_state` seed, so we can exactly reproduce the split later if we want to
- Randomize the row order with `shuffle=True`
- We use `stratify=target` to keep the distribution close to our target data
```
X_train, X_test, y_train, y_test = train_test_split(penguins_features, target, test_size=0.2, random_state=0, shuffle=True, stratify=target)
```
- For very large datasets your test_size can be much smaller
- Let's have a look at the dataset sizes
```
X_train.shape
```
```
X_test.shape
```
- Is our dataset well balanced? We can add up all categories, since they each have a value of `1`
```
y_train.sum()
```
- We see that there are many "Adelie", a bit less "Gentoo" penguins and even less "Chinstrap" penguins present in the dataset. This imbalance is already present in the original full dataset. Try to not have too strong imbalances in your datasets.

### 🛠️ Build an Architecture from Scratch (or Choose a Pretrained Model)
```
from tensorflow import keras
```
- Lets again make the process reproducible by setting a seed
```
from numpy.random import seed
seed(0)
```
```
from tensorflow.random import set_seed
set_seed(2)
```
- We use one hidden "dense" (or fully connected) layer
  - Dense means that all the input features are connected to the hidden layer neurons and all the hidden layer neurons are connected to the output features
- We get the Input layer shape directly from our dataset
```
inputs = keras.Input(shape=X_train.shape[1])
```
- We create a hidden layer with 10 neurons and the [REctified Linear Unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) "relu" activation function
  - For fully connected layers, choose a number larger than the number of input features
```
hidden_layer = keras.layers.Dense(10, activation="relu")(inputs)
```
- We want to predict one out of three species with the [Softmax](https://en.wikipedia.org/wiki/Softmax_function) "softmax" activation function
```
output_layer = keras.layers.Dense(3, activation="softmax")(hidden_layer)
```

#### Build the Model
```
model = keras.Model(inputs=inputs, outputs=output_layer)
```
```
model.summary()
```


## 📚 Resources
- Visualize and play with neurons (nodes) and layers on https://playground.tensorflow.org
- Explainable AI Library (DIANNA): https://github.com/dianna-ai/dianna
- Solve FizzBuzz with deep learning (please don't): https://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/
- Models on the net:
    - www.huggingface.co
    - https://modelzoo.co/
    - https://www.tensorflow.org/resources/models-datasets
    - https://github.com/onnx/models
- [Activation functions](https://en.wikipedia.org/wiki/Activation_function)