import pickle
import re

# Load
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def advanced_rules(msg):
    msg = msg.lower()

    if re.search(r'http[s]?://', msg):
        return True
    if re.search(r'\d+₹|\d+rs|\$\d+', msg):
        return True
    if any(word in msg for word in ["urgent", "act now", "limited", "offer", "click", "win"]):
        return True

    return False

# Input
message = input("Enter message: ")

# Rule check
if advanced_rules(message):
    print("SPAM ❌ (Rule-based)")
else:
    vector = vectorizer.transform([message])
    prob = model.predict_proba(vector)[0][1]

    if prob > 0.5:
        print(f"SPAM ❌ ({prob*100:.2f}%)")
    else:
        print(f"NOT SPAM ✅ ({(1-prob)*100:.2f}%)")