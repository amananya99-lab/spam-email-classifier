from flask import Flask, request, render_template
import pickle
import re

app = Flask(__name__)

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

@app.route('/')
def home():
    return render_template("index.html",
                           user_input="",
                           prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form.get('message', '')

    if advanced_rules(message):
        result = "SPAM ❌"
    else:
        vector = vectorizer.transform([message])
        prob = model.predict_proba(vector)[0][1]
        result = "SPAM ❌" if prob > 0.5 else "NOT SPAM ✅"

    return render_template("index.html",
                           user_input=message,
                           prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)