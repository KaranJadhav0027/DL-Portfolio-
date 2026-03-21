# Sentiment Analysis with SimpleRNN

A simple **Python-based Sentiment Analysis project** that demonstrates
how to build and train a Recurrent Neural Network (RNN) using
**TensorFlow/Keras**. This project classifies short text sentences into
**positive** or **negative** sentiments.

------------------------------------------------------------------------

## 📌 Features

-   Tokenization of text into integer sequences\
-   Padding for uniform input length\
-   Embedding layer for word vector representation\
-   SimpleRNN layer for sequential learning\
-   Dense layer with sigmoid activation for binary classification\
-   Trains on a tiny custom dataset (5 positive + 5 negative sentences)\
-   Tests model on unseen examples with sentiment prediction

------------------------------------------------------------------------

## 🏗️ Tech Stack

-   **Language**: Python\
-   **Libraries**: TensorFlow/Keras, NumPy

------------------------------------------------------------------------

## 📂 Dataset

A tiny dataset of 10 sentences:\
- **Positive (1):** "I love this movie", "This film was great", "What a
fantastic experience"...\
- **Negative (0):** "I hate this movie", "This film was terrible",
"Absolutely horrible acting"...

Balanced dataset → avoids class imbalance issues.

------------------------------------------------------------------------

## 🔧 How It Works

1.  **Tokenization** -- converts text to integer sequences.\
2.  **Padding** -- ensures fixed-length sequences (`maxlen=5`).\
3.  **Model Architecture**
    -   Embedding layer (word vectors)\
    -   SimpleRNN layer (8 hidden units, tanh)\
    -   Dense layer (sigmoid for binary classification)\
4.  **Training** -- optimizer = Adam, loss = binary crossentropy, 30
    epochs.\
5.  **Inference** -- test with new sentences, outputs "Positive" or
    "Negative".

------------------------------------------------------------------------

## 📊 Example Output

``` bash
Sentence: I enjoyed this film -> Sentiment: Positive
Sentence: I hated this film   -> Sentiment: Negative
```

------------------------------------------------------------------------

## 🚀 Run the Project

Clone this repository:

``` bash
git clone https://github.com/username/SimpleRNN-SentimentAnalysis.git
cd SimpleRNN-SentimentAnalysis
```

Install dependencies:

``` bash
pip install tensorflow numpy
```

Run the script:

``` bash
python sentiment_rnn.py
```

------------------------------------------------------------------------

## 📘 Learning Outcomes

-   Basics of **NLP preprocessing** (tokenization & padding)\
-   Building a **SimpleRNN model** in Keras\
-   Understanding **embedding layers** and sequence learning\
-   Applying binary classification with sigmoid activation

## 👨‍💻 Author

Developed by **Swamit Amit Jadhav**
📧 Contact: swamitaajadhav98@gmail(mailto:swamitaajadhav98@gmail.com)
