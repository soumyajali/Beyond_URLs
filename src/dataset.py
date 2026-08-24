import torch
from torch.utils.data import Dataset
import pandas as pd

class PhishingDataset(Dataset):
    """
    PyTorch Dataset wrapper for the Financial Phishing data.
    Takes tokenized inputs and their corresponding labels.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def generate_dummy_data(num_samples=1000):
    """
    Generates a robust mock dataset. 
    Crucially includes Legitimate financial transactions to ensure the model 
    doesn't blindly classify 'payment' or '₹' as phishing.
    """
    import random
    
    legitimate_templates = [
        # Normal conversations
        "Hey, are we still on for the meeting tomorrow?",
        "Don't forget to submit the project report by 5 PM.",
        "Can you send me the presentation slides?",
        "Happy birthday! Hope you have a great day.",
        # Legitimate Financial (CRITICAL TO PREVENT FALSE POSITIVES)
        "Your payment of ₹{amount} has been successfully processed. Transaction reference number is TXN{id}.",
        "Your monthly account statement is now available in your official banking application. Please log in to view your transactions.",
        "Your salary of ₹{salary} has been credited to your account for the month.",
        "Payment reminder: Your electricity bill of ₹{amount} is due on {date}.",
        "Dear Customer, your fund transfer of ₹{amount} to account ending in {acc} is successful.",
        "Amazon: Your order #{id} has been shipped and will arrive by tomorrow."
    ]
    
    suspicious_templates = [
        "Your account requires verification to continue using the service. Please review your account information.",
        "We noticed unusual activity associated with your account. Please verify your information to ensure uninterrupted service.",
        "Update your security settings immediately to prevent unauthorized access.",
        "Important notification regarding your delivery. It could not be delivered due to an invalid address.",
        "Your password will expire in 2 days. Consider updating it."
    ]
    
    deceptive_templates = [
        "URGENT! Your bank account will be suspended today. Send the OTP received on your phone to the verification officer immediately.",
        "Your refund of ₹{amount} is ready to be processed. Pay the ₹499 verification fee immediately to receive your refund.",
        "This is the bank security department. Your account has been flagged. Provide your OTP and account verification code immediately to prevent suspension.",
        "Urgent: I need you to buy 5 Apple gift cards for a client presentation. I will reimburse you later. - CEO",
        "Your bank account is compromised. Please wire ${amount} to the following safe account to protect your funds.",
        "Final notice: Pay your overdue tax bill of ₹{amount} immediately or you will be arrested.",
        "Send your OTP immediately to verify your bank account."
    ]
    
    texts = []
    labels = []
    
    for _ in range(num_samples):
        label = random.choices([0, 1, 2], weights=[0.4, 0.2, 0.4])[0] # Balanced distribution
        
        amount = random.randint(100, 10000)
        salary = random.randint(20000, 90000)
        txn_id = random.randint(100000, 999999)
        acc_end = random.randint(1000, 9999)
        date = f"{random.randint(1,28)}th Aug"
        
        if label == 0:
            text = random.choice(legitimate_templates)
        elif label == 1:
            text = random.choice(suspicious_templates)
        else:
            text = random.choice(deceptive_templates)
            
        # Format variables if they exist in the template
        text = text.replace('{amount}', str(amount))\
                   .replace('{salary}', str(salary))\
                   .replace('{id}', str(txn_id))\
                   .replace('{acc}', str(acc_end))\
                   .replace('{date}', date)
                   
        texts.append(text)
        labels.append(label)
        
    return pd.DataFrame({'text': texts, 'label': labels})
