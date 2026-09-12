# Amazon Customer Review : Sentiment Classification

Sentiment classification (Positive/Negative) on Amazon product reviews using TF-IDF and classical machine learning models.

## Approach

- Removed neutral (3-star) reviews; labeled 4-5 stars as Positive, 1-2 stars as Negative
- Excluded doRecommend, numHelpful, and username as features. Each checked and excluded for a specific, evidenced reason (redundancy, post-publication leakage, and memorization risk respectively), verified on train-only data
- Cleaned review text (lowercasing, stopword removal, negation preserved)
- Vectorized with TF-IDF
- Trained Logistic Regression, Naive Bayes, and Linear SVM, each tuned with GridSearchCV, evaluated by F1 score on the Negative class due to severe class imbalance (~94% Positive / ~6% Negative)
- Validated the TF-IDF choice against Word2Vec as an alternative feature extraction method after tuning

## Result

Logistic Regression and Linear SVM performed almost identically (F1 0.72 - 0.73 on the Negative class), a near-tie rather than a clear winner. Naive Bayes lagged behind (F1 0.54) due to having no built-in class-imbalance handling.

TF-IDF outperformed Word2Vec clearly (F1 0.72 vs 0.43) : Word-frequency weighting captured the specific words tied to negative reviews more directly than averaged word embeddings did.

## Installation

pip install -r requirements.txt

## Running

Open Amazon_Consumer_Reviews.ipynb in Google Colab, run all cells in order.

## Dataset

Datafiniti's Amazon Consumer Reviews of Amazon Products
Source: https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products
