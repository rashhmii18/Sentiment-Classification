# Sentiment Classification using NLP

## Project Overview

This project focuses on classifying text into two sentiment categories: Positive and Negative. Natural Language Processing (NLP) techniques and machine learning models were used to process the text, extract useful features, and perform sentiment classification.

The project also includes a Streamlit prototype that allows users to enter a sentence or review and receive a sentiment prediction.

## Dataset

Dataset Name: Sentiment Labelled Sentences

Source: UCI Machine Learning Repository

Dataset Link: https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences

The dataset contains sentiment-labelled sentences collected from three sources:

* Amazon
* IMDb
* Yelp

The sentiment labels are:

* 0 → Negative
* 1 → Positive

After combining the three files and removing duplicate records, the dataset used in this project contained 2,731 unique sentences.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* NLTK
* Sentence Transformers
* Joblib
* Streamlit

## Project Workflow

1. Dataset loading
2. Data inspection
3. Missing-value checking
4. Duplicate removal
5. Exploratory Data Analysis
6. Text cleaning
7. Stopword removal
8. Train-test splitting
9. TF-IDF feature extraction
10. Machine learning model training
11. Hyperparameter tuning
12. Pretrained sentence embeddings
13. Model evaluation
14. Error analysis
15. Model saving
16. Streamlit prototype development

## Data Preprocessing

The text data was cleaned before model training.

The preprocessing steps included:

* Converting text to lowercase
* Removing HTML tags
* Removing unnecessary special characters
* Removing extra whitespace
* Removing common stopwords
* Retaining important negation words such as 'no', 'not', and 'never'

Duplicate records were also removed from the dataset.

## Exploratory Data Analysis

The dataset was analysed using:

* Sentiment distribution
* Text length distribution
* Sentiment distribution by source
* Text length comparison between sentiment classes

The sentiment classes were almost evenly distributed, with approximately 50.38% positive and 49.62% negative sentences.

The three sources also contributed a similar number of records, so no single source dominated the dataset.

## TF-IDF Feature Extraction

TF-IDF was used to convert text into numerical features that machine learning models could process.

The baseline TF-IDF representation generated 4,304 features from the training data.

An additional TF-IDF experiment using unigrams and bigrams was also performed. The bigram representation did not provide an overall improvement compared with the unigram representation.

## Machine Learning Models

The following models were evaluated using TF-IDF features:

* Logistic Regression
* Linear SVM
* Multinomial Naive Bayes

Logistic Regression was also tuned using GridSearchCV with different values of the regularization parameter 'C'.

## Pretrained Sentence Embeddings

A pretrained all-MiniLM-L6-v2 sentence transformer was used to generate semantic representations of the sentences.

Each sentence was converted into a 384-dimensional embedding.

These embeddings were then provided to a Logistic Regression classifier.

The sentence embedding approach produced better results than the traditional TF-IDF representation, showing that semantic representations were useful for this sentiment classification task.

## Model Performance

| Model                                     | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ----------------------------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression - TF-IDF              |   80.80% |    81.09% | 80.80% |   80.94% |  88.53% |
| Logistic Regression - TF-IDF Tuned        |   80.62% |    80.36% | 81.52% |   80.94% |  88.69% |
| Linear SVM - TF-IDF                       |   80.44% |    79.44% | 82.61% |   80.99% |       — |
| Multinomial Naive Bayes - TF-IDF          |   79.52% |    77.89% | 82.97% |   80.35% |       — |
| Logistic Regression - Sentence Embeddings |   84.83% |    85.61% | 84.06% |   84.83% |  93.51% |

## Final Model

The final model uses:

* all-MiniLM-L6-v2 for sentence embeddings
* Logistic Regression for sentiment classification

The final model achieved:

* Accuracy: 84.83%
* Precision: 85.61%
* Recall: 84.06%
* F1-score: 84.83%
* ROC-AUC: 93.51%

Compared with the baseline TF-IDF Logistic Regression model, the sentence embedding approach improved accuracy by about 4 percentage points and ROC-AUC by about 5 percentage points.

## Error Analysis

The final model still made some incorrect predictions on the test data.

Some difficult examples involved:

* Negation
* Sarcasm
* Context-dependent expressions
* Indirect sentiment
* Numerical ratings

These cases can be challenging for traditional machine learning classifiers because their meaning can depend on context.

## Streamlit Prototype

A separate Streamlit application was created to demonstrate the trained model.

The application allows users to:

1. Enter a sentence or review.
2. Generate a sentence embedding using all-MiniLM-L6-v2.
3. Predict the sentiment using the trained Logistic Regression model.
4. Display the result as Positive Sentiment or Negative Sentiment.

## Project Structure

Sentiment-Classification/

├── app.py
├── sentiment_embedding_lr.pkl
├── Sentiment_Classification.ipynb
└── README.md

## How to Run the Prototype

### 1. Install the required libraries

pip install streamlit sentence-transformers joblib

### 2. Run the Streamlit application

python -m streamlit run app.py

The application will open in a local web browser.

## Files Description

Sentiment_Classification.ipynb
Contains the complete data analysis, preprocessing, model training, evaluation, and experimentation.

sentiment_embedding_lr.pkl
Saved Logistic Regression model trained using sentence embeddings.

app.py
Streamlit application used for the sentiment prediction prototype.

README.md
Project documentation.

## Conclusion

This project demonstrated a complete NLP sentiment classification workflow, from data preprocessing and exploratory analysis to model training, evaluation, and deployment through a Streamlit prototype.

Traditional TF-IDF-based machine learning models achieved around 80% accuracy, while the pretrained sentence embedding approach achieved 84.83% accuracy and 93.51% ROC-AUC. The results show that pretrained sentence embeddings provided a more effective representation of the text for this dataset.

The project also demonstrates how a trained NLP model can be saved and integrated into a simple interactive application for sentiment prediction.
