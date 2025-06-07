

Toxic Comment Classification Using Deep Learning
Project Overview
This project implements and compares deep learning models for detecting toxic comments in online discussions. The goal is to classify comments across six categories of toxicity: toxic, severe_toxic, obscene, threat, insult, and identity_hate.
Dataset
The project uses the Jigsaw Toxic Comment Classification Challenge dataset from Kaggle, which contains:

Training set: 159,571 comments with binary labels for toxicity categories
Test set: 153,164 comments for prediction
Highly imbalanced dataset with most comments being non-toxic

Models Implemented
1. LSTM (Long Short-Term Memory)

Embedding layer: 20,000 vocabulary, 128 dimensions
LSTM layer: 64 units with dropout regularization
Global max pooling and dense layers
Final sigmoid activation for multi-label classification

2. CNN (Convolutional Neural Network)

Same embedding configuration as LSTM
1D Convolutional layer with 64 filters
Global max pooling for feature extraction
Dense layers for classification

3. Hyperparameter Tuning
Tested variations of the winning LSTM model:

Bigger network (150 embedding dims, 80 LSTM units)
Lower learning rate (0.0005 vs 0.001)
Systematic comparison using AUC scores

Methodology
Data Preprocessing

Text lowercasing and basic cleaning
Tokenization with 20,000 most frequent words
Sequence padding to length 200
Train/validation split (90/10)

Model Training

Binary crossentropy loss for multi-label classification
Adam optimizer with learning rate scheduling
Early stopping and model checkpointing
Batch size: 128, trained for multiple epochs

Evaluation

Primary metric: Area Under ROC Curve (AUC)
Secondary metric: Accuracy
AUC chosen due to class imbalance in dataset
