import pickle

tokenizer = pickle.load(open("models/cv.pkl", "rb"))
model = pickle.load(open("models/clf.pkl", "rb"))

def get_preds(email_text: str):
    tokenized_email = tokenizer.transform([email_text])
    predictions = model.predict(tokenized_email)
    predictions = 1 if predictions == 1 else -1
    return predictions