# NLP-Sentiment-Analysis-Mental-Health
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ManuelEspejo/NLP-Sentiment-Analysis-Mental-Health/blob/main/text-classification-with-neural-networks.ipynb)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=101010)](https://www.python.org/)

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white&labelColor=101010)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.10.0-D00000?style=for-the-badge&logo=keras&logoColor=white&labelColor=101010)](https://keras.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-1.2.2-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white&labelColor=101010)](https://scikit-learn.org/stable/#)

[![NumPy](https://img.shields.io/badge/NumPy-1.23.5-013243?style=for-the-badge&logo=numpy&logoColor=white&labelColor=101010)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.5.3-150458?style=for-the-badge&logo=pandas&logoColor=white&labelColor=101010)](https://pandas.pydata.org/docs/#)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.1-3776AB?style=for-the-badge&labelColor=101010)](https://matplotlib.org/stable/index.html#)

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Models](#models)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

The objective of this project is to demonstrate binary classification using various algorithms and explore their performance in determining whether short text messages are related to mental health or not. Additionally, we aim to gain insights into the types of mistakes made by these models. We leverage tools such as confusion matrices to analyze these errors and evaluate the effectiveness of our models.

## Installation

Clone this repository and install the required dependencies using the following command:
```bash
git clone https://github.com/ManuelEspejo/NLP-Sentiment-Analysis-Mental-Health.git
```
Or using the [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ManuelEspejo/NLP-Sentiment-Analysis-Mental-Health/blob/main/text-classification-with-neural-networks.ipynb) button.

## Usage

1. Open the `sentiment_analysis.ipynb` Jupyter Notebook.
2. Follow the step-by-step instructions provided in the notebook to perform the analysis.
3. Modify the code or experiment with different models and techniques to enhance the analysis.
4. Execute the notebook cells to see the results and insights.

## Dataset

The dataset used in this project consists of short text messages labeled as either mental health-related (1) or not considered (0). The dataset has been **downloaded from kaggle**. You can download it from [here](https://www.kaggle.com/datasets/reihanenamdari/mental-health-corpus).

## Models

We experiment with various machine learning models, including **Dense model**, **Convolutional Neural Networks** and **LSTM**, to perform sentiment analysis. The notebook explores the implementation and comparison of these models for the **text binary classification task**.

## Evaluation

To evaluate the performance of the models, we use common evaluation metrics such as accuracy, precision, recall, and F1-score. 

## Results

The results of the sentiment analysis demonstrate the effectiveness of the applied machine learning models. We discuss the achieved accuracy and provide visualizations of evaluation metrics for a comprehensive understanding of the analysis.

## Contributing

Contributions to this project are welcome. If you have any suggestions or would like to enhance the analysis, please feel free to submit a pull request.

At the end of the notebook there is also a "next steps" section, in wich I have talked about some approaches that we could take with this notebook for further improve our models performance. 
Anyone is welcome to try one of those approaches to see the potential improvements, it would be great to see that.

[![GitHub Issues](https://img.shields.io/badge/github-issues-F36D5D?style=for-the-badge&logo=github&logoColor=white&labelColor=101010)](https://github.com/ManuelEspejo/NLP-Sentiment-Analysis-Mental-Health/issues)
[![GitHub Pull Request](https://img.shields.io/badge/github-Pull_Request-F8DC75?style=for-the-badge&logo=github&logoColor=white&labelColor=101010)](https://github.com/ManuelEspejo/NLP-Sentiment-Analysis-Mental-Health/pulls)

## License

This project is licensed under the MIT License.

[![MIT License](https://img.shields.io/badge/MIT-License-brightgreen?style=flat&labelColor=101010)](https://github.com/ManuelEspejo/NLP-Sentiment-Analysis-Mental-Health/blob/main/LICENSE)


