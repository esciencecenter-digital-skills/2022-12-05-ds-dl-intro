![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document Day 2

2022-12-05-ds-dl-intro

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: [link](https://tinyurl.com/2022-12-05-ds-dl-intro-day2)

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
| 9:00 | 	Recap |
| 9:15 | Classification by a neural network, continued from step 5  |
| 10:15 | Coffee break |
| 10:30 | Classification by a neural network, continued|
|11:30 | Tea break |
|11:45 | Monitoring the training process|
|12:45 | Wrapping up|
|13:00 | End |



## 🧠 Collaborative Notes

### Step 5: :hole: Choose a loss function and optimizer
```python=
model.compile(loss = keras.losses.CategoricalCrossentropy(), optimizer = 'adam')
```
### Step 6: 🚂 Train model
```python=
history = model.fit(X_train, y_train, epochs=100)
```

Let's plot the loss:
```python=
sns.lineplot(x=history.epoch, y=history.history['loss'])
```

### Step 7: 🔮 Perform a prediction/classification
```python=
y_pred = model.predict(X_test)
prediction = pd.DataFrame(y_pred, columns=target.columns)
prediction
```

```python=
predicted_species = prediction.idxmax(axis='columns')
predicted_species
```

### Step 8: 🏁 Measure performance
We have trained a neural network, and we now want to check how it performs. We will do this using a confusion matrix.
```python=
from sklearn.metrics import confusion_matrix

true_species = y_test.idxmax(axis="columns")

matrix = confusion_matrix(true_species, predicted_species)
print(matrix)
```

```python=
# Convert to a pandas dataframe
confusion_df = pd.DataFrame(matrix, index=y_test.columns.values, columns=y_test.columns.values)

# Set the names of the x and y axis
confusion_df.index.name = 'True Label'
confusion_df.columns.name = 'Predicted Label'

sns.heatmap(confusion_df, annot=True)
```

### Step 9: 🛠️ Tune the model/hyperparameters
What can we modify in the model to improve its performance? (Exercise "Trial and error: play with your model")

### Step 10: 🤝 Save/share model
```python=
model.save('my_first_model')
```

```python=
pretrained_model = keras.models.load_model('my_first_model')
```

```python=
y_pretrained_pred = pretrained_model.predict(X_test)
pretrained_prediction = pd.DataFrame(y_pretrained_pred, columns=target.columns.values)
```

```python=
pretrained_predicted_species = pretrained_prediction.idxmax(axis='columns')
pretrained_predicted_species
```

### Episode 3: 🧐 Monitor the training process
Open a new Jupyter notebook.

Download data if needed:
```python=
data = pd.read_csv("https://zenodo.org/record/5071376/files/weather_prediction_dataset_light.csv?download=1")
```

#### 1. Outline the problem: weather prediction
Goal: Predict tomorrow's sunshine in BASEL city

#### 2. Identify inputs and outputs (data exploration)
```python=
# Import the dataset

import pandas as pd
```

```python=
filename_data = 'weather_prediction_dataset_light.csv'
```

```python=
data = pd.read_csv(filename_data)
```

```python=
data.head()
```

```python=
data.columns
```

```python=
import string
measurements = set()
for x in data.columns:
    if x not in ["DATE", "MONTH"]:
        measure = x.lstrip(string.ascii_uppercase + '_')
        measurements.add(measure)
```

```python=
print(measurements)
```

```python=
len(measurements)
```

```python=
data.iloc[:365]['BASEL_sunshine'].plot(xlabel='Day', ylabel='BASEL sunshine hours')
```

#### 3. Prepare data
Use 3 years of data:
```python=
nr_rows = 365 * 3
```

```python=
X_data = data.loc[:nr_rows].drop(columns=['DATE', 'MONTH'])
```

```python=
X_data.head()
```

```python=
X_data.shape
```

```python=
y_data = data.loc[1:(nr_rows+1)]['BASEL_sunshine']
y_data.shape
```

```python=
y_data.head()
```

#### Split data into training, validation and test sets
```python=
from sklearn.model_selection import train_test_split
```

```python=
X_train, X_holdout, y_train, y_holdout = train_test_split(X_data, y_data, test_size=0.3, random_state=0)
```

```python=
X_val, X_test, y_val, y_test = train_test_split(X_holdout, y_holdout, test_size=0.5, random_state=0)
```

```python=
X_train.shape
```

```python=
X_val.shape
```

```python=
y_train.shape
```

```python=
y_val.shape
```

```python=
y_test.shape
```

#### 4. Build architecture or use a pretrained model
See exercise: "Architecture of the Network"

```python=
from tensorflow import keras
```

```python=
def create_nn():
    # input layer
    inputs = keras.Input(shape=(X_train.shape[1],), name='input')
    
    # Dense layers
    layers_dense = keras.layers.Dense(100, 'relu')(inputs)
    layers_dense = keras.layers.Dense(50, 'relu')(layers_dense)
    
    # output layer
    outputs = keras.layers.Dense(1)(layers_dense)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='weather_prediction_model')
    return model
```

```python=
model = create_nn()
```

```python=
model.summary()
```

## 📚 Resources
- Loss functions: https://www.tensorflow.org/api_docs/python/tf/keras/losses