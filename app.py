import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the tokenizer and model from the model folder
tokenizer = AutoTokenizer.from_pretrained("./model")
model = AutoModelForSequenceClassification.from_pretrained("./model", trust_remote_code=True)

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

# Create Gradio interface
iface = gr.Interface(
    fn=predict_complaint_topic,  # Function to call for prediction
    inputs=gr.Textbox(label="Enter your complaint text"),  # Input type (Textbox)
    outputs=gr.Textbox(label="Predicted Complaint Topic"),  # Output type (Textbox)
    live=True,  # Enable live prediction as the user types
    title="Complaint Topic Classifier",  # Title of the app
    description="This model classifies complaints into different topics like 'Bank Account Services', 'Credit Card or Prepaid Card', etc."  # Description of the app
)

# Launch the interface
iface.launch()
