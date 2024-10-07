# Text-Sentiment-Classification

## Abstract
This project introduces an approach for automatically classifying the sentiment of Twitter messages as either positive or negative. This is particularly useful for consumers wanting to research the sentiment of products before purchase, or companies that monitor the public sentiment towards their brands.

The classification is performed using machine learning algorithms trained on a dataset of tweets that are labeled based on the presence of emoticons, which serve as noisy labels. The dataset consists of large-scale Twitter data, allowing for supervised learning using algorithms like Naive Bayes, Maximum Entropy, and SVM, achieving over 80% accuracy. This repository replicates the classification process using modern techniques.

For more details on the paper that introduces this approach, refer to the original work:
- [Paper](https://www-cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf)

The dataset used is publicly available:
- [Sentiment140 Dataset on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

---

## Assessment

This project is part of an intermediate AI exam focused on understanding and improving text classification models by analyzing a labeled text dataset. The objective is to:

1. **Choose a labeled text dataset**: The dataset used is Sentiment140 from Kaggle, which classifies tweets as either positive or negative. The dataset is chosen due to its clear business application in sentiment analysis.
2. **Define a classification pipeline**: The project involves creating a pipeline that includes pre-processing steps like lemmatization, stopword removal, and text vectorization using techniques like CountVectorizer and TfidfVectorizer. We also explore stemming, lemmatization, and feature engineering.
3. **Run and evaluate classifiers**: Multiple classifiers such as Naive Bayes and Logistic Regression are used. Evaluation is performed using diverse train-test shuffles and metrics like accuracy and balanced accuracy score. The importance of specific words in classification decisions is analyzed.
4. **Dataset size assessment**: The accuracy and errors are evaluated over different dataset sizes to analyze how increasing data impacts performance. This includes evaluating if adding more data improves results and if it's feasible within the business context.
5. **Topic models**: Finally, topic models are used to evaluate classification performance over different topics. A two-layer classifier is tested, where the first layer classifies the topic, and the second layer focuses on sentiment classification within each topic.

---

## Project Structure

- **`main.ipynb`**: This Jupyter Notebook contains all the code for data preprocessing, classifier pipelines, evaluation, and topic modeling. Each section of the assessment is implemented within this notebook, making it easy to follow along with the classification process.

- **`Report.pdf`**: PDF containing methods and results.

- **`data/`**: The dataset files used for training and evaluation, including the processed Sentiment140 dataset.

- **`results/`**: Directory containing evaluation results, such as accuracy scores, confusion matrices, and learning curves.

