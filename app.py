import streamlit as st
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("distilbert_complaint_classifier_final")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert_complaint_classifier_final")

# Set the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define topic mapping
topic_mapping = {
    0: 'Bank Account Services',
    1: 'Credit Card or Prepaid Card',
    2: 'Others',
    3: 'Theft/Dispute Reporting',
    4: 'Mortgage/Loan'
}

# Prediction function
def predict_complaint_topic(complaint_text):
    encoding = tokenizer.encode_plus(
        complaint_text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
    
    predicted_topic = topic_mapping[predicted_class_id]
    return predicted_topic

# Streamlit UI setup
st.title("Complaint Topic Classifier")
st.write("This model classifies complaints into different topics like 'Bank Account Services', 'Credit Card or Prepaid Card', etc.")

# Input for complaint text
complaint_text = st.text_area("Enter your complaint text")

# Predict button
if st.button("Predict"):
    if complaint_text:
        predicted_topic = predict_complaint_topic(complaint_text)
        st.subheader("Predicted Complaint Topic")
        st.write(predicted_topic)
    else:
        st.warning("Please enter a complaint text.")
