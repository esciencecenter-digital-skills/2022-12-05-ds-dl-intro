![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document Day 4

2022-12-05-ds-dl-intro

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: [link](https://tinyurl.com/2022-12-05-ds-dl-intro-day4)

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
| 9:00 | Recap, discuss last question chapter 3, tensorboard |
| 9:15 | Advanced layer types|
| 10:15 | Coffee break |
| 10:30 |Advanced layer types|
|11:30 | Tea break |
|11:45 | Advanced layer types|
|12:45 | Wrapping up|
|13:00 | End |


## 🧠 Collaborative Notes

Demo of tensorboard (don't need to type along for this part)
```python=
from tensorflow.keras.callbacks import TensorBoard
import datetime

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

model = create_nn_batchnorm()
compile_model(model)

history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=200,
                    validation_data=(X_val, y_val),
                    callbacks=[tensorboard_callback],
                    verbose=2)
```

```python=
%load_ext tensorboard
%tensorboard --logdir logs/fit
```

### Advanced layer types
#### Goal: Image classification using CIFAR-10 dataset (10 classes)

#### Data exploration
```python=
from tensorflow import keras
```

```python=
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
```

```python=
n = 5000
train_images = train_images[:n]
train_labels = train_labels[:n]
```

```python=
type(train_images)
```

```python=
train_images.shape
```

```python=
train_images.max()
```

```python=
train_images.min()
```

```python=
train_labels.shape
```

```python=
train_labels.min()
```

```python=
train_labels.max()
```

```python=
train_images = train_images/255.0
test_images = test_images/255.0
```

```python=
train_images.max()
```

```python=
train_images.min()
```

```python=
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
```

```python=
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.axis('off')
    plt.title(class_names[train_labels[i,0]])
plt.show()
```

```python=
dim = train_images.shape[1] * train_images.shape[2] * train_images.shape[3]
```


### Build an architecture
```python=
inputs = keras.Input(shape=train_images.shape[1:])
```

```python=
# hidden layers
x = keras.layers.Conv2D(50, (3,3), activation='relu')(inputs)
x = keras.layers.MaxPooling2D((2,2))(x)
x = keras.layers.Conv2D(50, (3,3), activation='relu')(x)
x = keras.layers.MaxPooling2D((2,2))(x)
```

```python=
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(50, activation='relu')(x)
```

```python=
# output layer
outputs = keras.layers.Dense(10)(x)
```

```python=
# model
model = keras.Model(inputs=inputs, outputs=outputs, name='cifar_model_small')
```

```python=
model.summary()
```

```python=
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

```python=
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
```

```python=
import seaborn as sns
import pandas as pd
```

```python=
history_df = pd.DataFrame.from_dict(history.history)
```

```python=
sns.lineplot(data=history_df[['accuracy', 'val_accuracy']])
```

```python=
sns.lineplot(data=history_df[['loss', 'val_loss']])
```

Time to talk about dropout layers.

Dropout rate: A value between 0 and 1
```python=
def create_nn():
    inputs =     keras.Input(shape=train_images.shape[1:])
    x = keras.layers.Conv2D(32, (3, 3),     activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = keras.layers.Dropout(0.8)(x)
    x = keras.layers.Flatten()(x)
outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs=inputs,     outputs=outputs, name="cifar_model_small")
```

```python=
def compile_model(model):
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
```

```python=
model = create_nn()
compile_model(model)
model.summary()
```

```python=
history_dropout = model.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))
```

```python=
history_dropout_df = pd.DataFrame.from_dict(history_dropout.history)
sns.lineplot(data=history_dropout_df[['accuracy', 'val_accuracy']])

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
```

```python=
sns.lineplot(data=history_dropout_df[['loss', 'val_loss']])
```


## 👍👎 Tips/Tops
### 🔝 What went well? (top)
- reached my personal goal: more understanding of deep learning and the parameters that have to be tweaked. Thank you!
- Happy that we did some basic image classification, as the classification and regression examples were quite simple (also they are more suited for basic machine learning algorithms)
- Actually I liked the HackMD interface (despite the glitches)
- Answered all my questions!
- I found it really useful to work with different data types (images, weather data)
- I am going to steal the state-of-the-art image classifier and try to get around captchas that my scrapers are always bumping into.
- Very useful explanation, good level for me.
- I found Cunliangs explanations really good! Of course also Djura, but today Cunliang stole the show! =D
- The content is great and the flow from the instructors. also the help/
- Cunliang cleared the eternal fog that surrounds CNNs for me today quite a bit, cheers!
- I learned a lot , by diving in with limited python/pandas experience (but knowledge of other languages). Thanks
- Nice to use examples outside of image classification as this is where I want to apply DL

### 💪 What could we improve? (tip)

- Would be nice to have the equations for how the parameters are calculated, that section was a bit confusin
    - CG: see the resources below, I have [provided there](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks#)
- Would be nice to have an Intermediate Deep learning course
- I prefer the use of functions in the notebooks, rather than editing earlier cells and re-running everything -- I think it makes the resulting notebook much clearer if I come back to it later.
- to round of the story of classification of images it would be nice to end with images (prediction) :+1:
- maybe seperate the demo's and background information from each other. Now it is sometimes difficult to follow both at the same time. So e.g. alternate between powerpoint presentations and demo
- Adding a workflow of the steps that we are following at the top of this HackMD document i.e, a flowchart?
- I enjoyed working with different data types, maybe add textual data too. Narrative wise I would like to also work with a dataset where the performance actually improves with the additions (dropout/normalization/etc.) that are added. This way you end up with a better understanding of the methods to tackle overfitting.

- finish with this code, to make the result more tangible
``` python 
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.axis('off')
    plt.title(class_names[predictions[i].argmax()])
plt.show()
```
- ### Post workshop survey!
https://www.surveymonkey.com/r/RNJNRGP


## 📚 Resources
- [Hyperparameter tuning in tensorboard](https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams)
- [Play with convolution filter visually](https://setosa.io/ev/image-kernels/)
- [Stanford DL course summary](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks#)
    - it give equations on how to calculate number of parameters for different types of layers
- [SURF](https://www.surf.nl/)
- GPU setup
  - https://www.tensorflow.org/install/pip (Point 4)
  - Check if TensorFlow uses GPU
    `print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))`
- Keras [trainable and non-trainable parameters](https://stackoverflow.com/questions/47312219/what-is-the-definition-of-a-non-trainable-parameter)
    - in short, paramters are not updated using  gradients (calculated from loss) are non-trainable parameters.
- [state-of-the-art models for CIFAR-10 data](https://paperswithcode.com/sota/image-classification-on-cifar-10)
- [our songs](https://youtube.com/playlist?list=PLCCdkKPN_mNdSz4L6-KTXjMaSGCWJ-7S6)
## ⬆️📋⬇️Copy & Paste

### Breakout Room 1

### Breakout Room 2

### Breakout Room 3