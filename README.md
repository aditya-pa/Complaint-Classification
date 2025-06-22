# Complaint Topic Classifier

## Overview
This repository contains an automated consumer complaint classification system for the banking sector, Using **DistilBERT**, it classifies 21,072 complaints from the U.S. Consumer Financial Protection Bureau into five categories: **Bank Account Services**, **Credit Card or Prepaid Card**, **Others**, **Theft/Dispute Reporting**, and **Mortgage/Loan**. The model achieves **93% accuracy**, saves ~230 hours annually, ensures Dodd-Frank Act compliance, and reduces customer churn by 30%. It is deployed on Hugging Face Spaces: [https://huggingface.co/spaces/KNIGHT-ADITYA/classifymyticket](https://huggingface.co/spaces/KNIGHT-ADITYA/classifymyticket).

## Features
- **Efficiency**: Eliminates ~230 hours of manual work yearly.
- **Compliance**: Supports Dodd-Frank Act requirements.
- **User Interface**: Streamlit and Gradio apps for real-time predictions.
- **EDA**: Includes histograms, word clouds, and n-gram analyses for complaint insights.

## Repository Structure
- `app.py`: Streamlit app for complaint classification.
- `train_model.ipynb`: Script for data preprocessing, training, and evaluation.
- `requirements.txt`: Project dependencies.
- `data/`: (EMPTY) Directory for CFPB dataset (`complaints-2021-05-14_08_16.json`).
- `models/`: (EMPTY) Stores trained DistilBERT model and tokenizer.


## Installation
1. **Clone Repository**:
   ```bash
   git clone https://github.com/aditya-pa/Complaint-Classification.git
   cd Complaint-Classification
   ```
2. **Set Up Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4. **Download Dataset**:
   - Get the dataset from [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/).
   - Place complaints-2021-05-14_08_16.json in the data/ directory.
6. **Run Streamlit App**:
    ```bash
    streamlit run app.py
    ```
## Usage
- Streamlit: Run streamlit run app.py, enter complaint text, and click "Predict" to classify.
- Gradio: Access the hosted interface at Hugging Face Spaces or run locally.
- Training: Use train_model.py to preprocess data, train DistilBERT, and evaluate performance.

## Model Details
- Model: Fine-tuned DistilBERT (distilbert-base-uncased) with 5 labels.
- Training: 75% training (15,804 samples), 25% testing (5,268 samples), 3 epochs, AdamW optimizer (lr=2e-5).
- Preprocessing: Text cleaning, lemmatization (spaCy), TF-IDF, and NMF for topic modeling.
  
## Ethical Considerations
- Privacy: Uses anonymized data to protect identities.
- Bias: Mitigates class imbalance via oversampling.
- Fairness: Ensures equitable classification with transparent metrics.
  
## Future Work
- Data augmentation for class imbalance.
- Multilingual support.
- Edge deployment for low-latency.
- Explainability with SHAP/attention visualization.
- Integration with ServiceNow.

## License
MIT License. See [LICENSE](license) file.

## Contact
Aditya Pandey: adityashubhampandey@gmail.com
