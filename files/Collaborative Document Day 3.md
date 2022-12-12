![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document Day 3

2022-12-05-ds-dl-intro

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: [link](https://tinyurl.com/2022-12-05-ds-dl-intro-day3)

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
| 9:00 | recap |
| 9:15 | Monitor the training process, continued  |
| 10:15 | Coffee break |
| 10:30 |Monitor the training process, continued|
|11:30 | Tea break |
|11:45 | Tensorboard|
|12:45 | Wrapping up|
|13:00 | End |

## 🧠 Collaborative Notes
### 🔁 Recap Activation Function
- Introduces non-linearity
- Allows for solving more complex problems
- There is a set of proven [activation functions](https://en.wikipedia.org/wiki/Activation_function) to choose from
- The last layer usually should not have an activation function, otherwise you would confine your range of possible outputs
- It's recommended to stick to _one_ activation function for all layers

### :heavy_plus_sign: Addendum: Prepare Data
- Are there any null values?
```python=
#data.isnull()  # Boolean per value
#data.isnull().any()  # Boolean per variable
data.isnull().any().any()  # Boolean per whole set
```

### :hole: 5. Choose Loss Function and Optimizer
Loss function options:
- Mean Squared Error `mse`
  - Removes negatives and punishes large errors more than small ones
- Mean Absolute Error `mae`
  - "Only" removes negatives
  - Preserves units, intuitive

Optimizer options:
- Stochastic Gradient Decent
  - Fixed step size
- Adam
  - More intricate
  - Common choice

Metrics:
- Stats for you to follow
- No influence on the model
- Example: [Root Mean Square Error](https://keras.io/api/metrics/regression_metrics/#rootmeansquarederror-class)
  - The "root" brings the error back to intuitive units

### 🚂 6. Train the Model

```python=
def compile_model(model):
    model.compile(loss="mse",
                  optimizer="adam",
                  metrics=[keras.metrics.RootMeanSquaredError()])
```

```python=
compile_model(model)
```
Batch size:

- Large: more accurate gradient, but slower
- Small: quick 'n dirty

```python=
history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=200,
                    verbose=2)
```

```python=
history.history
```

Let's visualize the history.
```python=
import seaborn as sns
import matplotlib.pyplot as plt

def plot_history(history, metrics):
    history_df = pd.DataFrame.from_dict(history.history)
    sns.lineplot(data=history_df[metrics])
    
    plt.xlabel("epochs")
    plt.ylabel("metric")
```

```python=
plot_history(history, ["root_mean_squared_error"])
```
```python=
plot_history(history, ["root_mean_squared_error", "loss"])
```

### 🔮 7. Perform a Prediction

```python=
y_train_predicted = model.predict(X_train)
y_test_predicted = model.predict(X_test)
```

We can't use a confusion matrix for performance analysis, since we don't have categories. Instead, we can use a scatter plot.

```python=
def plot_predictions(y_pred, y_true, title):
    plt.style.use("ggplot")
    plt.scatter(y_pred,
                y_true,
                s=10,     # s: marker size
                alpha=.5) # alpha: opacity
    plt.xlabel('predicted sunshine hours')
    plt.ylabel('true sunshine hours')
    plt.title(title)
```

```python=
plot_predictions(y_train_predicted, y_train, title="Predictions on the training set")
```

```python=
plot_predictions(y_test_predicted, y_test, title="Predictions on the test set")
```
Why does the performance on the `test` set so bad? We might be overfitting.
![overfitting](https://www.datarobot.com/wp-content/uploads/2018/03/Screen-Shot-2018-03-22-at-11.22.15-AM-e1527613915658.png)

### Step 8: 🏁 Measure performance

Can we quantify the performance?
```python=
train_metrics = model.evaluate(X_train, y_train, return_dict=True)
test_metrics = model.evaluate(X_test, y_test, return_dict=True)

print("Train RMSE: {:.2f}, Test RMSE: {:.2f}".format(
    train_metrics["root_mean_squared_error"],
    test_metrics["root_mean_squared_error"]))
```
<!--We have not touched any hyperparameters yet.-->
How good or bad do we perform? It's handy to compare the results to a simple baseline model. Let's assume that tomorrow will have comparable sunshine hours as today. :sunny: 

```python=
y_baseline_prediction = X_test["BASEL_sunshine"]
plot_predictions(y_baseline_prediction,
                 y_test,
                 title="Baseline predictions on the test set")
```

```python=
from sklearn.metrics import mean_squared_error

rmse_baseline = mean_squared_error(y_test,
                                   y_baseline_prediction,
                                   squared=False)  # get the root instead
print("Baseline: ", rmse_baseline)
print("Neural network: ", test_metrics["root_mean_squared_error"])
```

### Step 9: 🛠️ Tune the model/hyperparameters

Neural networks are prone to overfitting. It's best to keep a _separate_ validation set. With this we can stop the training when signs of overfitting start to occur.

- N.B.: "test set" and "validation set" are sometimes used interchangably. No matter what you name your sets, use one during training, and one _after_.

We create a new, clean model and we will evaluate the performance during training.

```python=
model = create_nn()
compile_model(model)
```

```python=
history = fit(X_train, y_train,
             batch_size=32,
             epochs=200,
             validation_data=(X_val, y_val))
```
It's easier to interpret visually:
```python=
plot_history(history, ["root_mean_squared_error", "val_root_mean_squared_error"])
```

We can also look at the trained weights:
```python=
model.weights
```
```python=
for layer in model.layers:
    print(layer.get_config(), layer.get_weights())
```
Let's create a new model again to try more overfitting-countermeasures.
```python=
def create_nn(first_layer_size=100, second_layer_size=200):
    """New version with adjustable layer size.
    Old defaults are kept."""
    # input layer
    inputs = keras.Input(shape=(X_train.shape[1],), name="input")
    
    # dense layers
    layers_dense = keras.layers.Dense(first_layer_size, "relu")(inputs)
    layers_dense = keras.layers.Dense(second_layer_size, 'relu')(layers_dense)
    
    # output layer
    outputs = keras.layers.Dense(1)(layers_dense)
    
    model = keras.Model(inputs=inputs,
                        outputs=outputs,
                        name='weather_prediction_model')
    return model
```

```python=
model = create_nn()
compile_model(model)
```

```python=
from tensorflow.keras.callbacks import EarlyStopping

earlystopper = EarlyStopping(monitor = 'val_loss',
                             patience = 10) # wait a couple of epochs before taking action

history = model.fit(X_train,
                    y_train,
                    batch_size=32,
                    epochs=200,
                    validation_data=(X_val, y_val),
                    callbacks=[earlystopper])
```

How did it perform?
```python=
plot_history(history, ["root_mean_squared_error",
                       "val_root_mean_squared_error"])
```
Stopping early improves the results _and_ save resources. 💪
```python=
from tensorflow.keras.layers import BatchNormalization
```

Again, new model. This time we normalize layers, see https://keras.io/api/layers/normalization_layers/batch_normalization/
```python=
def create_nn_batchnorm(first_layer_size=100, second_layer_size=200):
    """New version with adjustable layer size.
    Old defaults are kept."""
    # input layer
    inputs = keras.Input(shape=(X_train.shape[1],), name="input")
    
    # batch normalization
    batchnorm = BatchNormalization()(inputs)
    
    # dense layers
    layers_dense = keras.layers.Dense(first_layer_size, "relu")(batchnorm)
    layers_dense = keras.layers.Dense(second_layer_size, 'relu')(layers_dense)
    
    # output layer
    outputs = keras.layers.Dense(1)(layers_dense)
    
    model = keras.Model(inputs=inputs,
                        outputs=outputs,
                        name='weather_prediction_model')
    return model
```

```python=
model = create_nn_batchnorm()
compile_model(model)
model.summary()
```

```python=
history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=1000,  # rediculously high, but we will stop early
                    validation_data=(X_val, y_val),
                    callbacks=[earlystopper])
```
Normalization helped with the performance!
```python=
plot_history(history, ["root_mean_squared_error",
                       "val_root_mean_squared_error"])
```

Finally, let's see how the model performs on the test set.
```python=
y_test_predicted = model.predict(X_test)
plot_predictions(y_test_predicted, y_test, title="Predictions on test set")
```
### Step 10: 🤝 Save/share model

```python=
model.save("my_tuned_weather_model")
```



## 📚 Resources
- Deep learning courses:
    - [Fast.ai](https://www.fast.ai/)
- [Adam optimizer](https://machinelearningjourney.com/index.php/2021/01/09/adam-optimizer/)
- [ReLU vs Sigmoid](https://medium.com/geekculture/relu-vs-sigmoid-5de5ff756d93)
- [Keras metrics](https://keras.io/api/metrics/)
- [Batch size - an experiment](https://androidkt.com/ideal-batch-size-for-the-keras-neural-network/)

