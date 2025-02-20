from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Initialize Flask app
app = Flask(__name__)

# Load Flan-T5 model & tokenizer
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Initialize pipeline
chatbot = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.json.get("question")
    response = chatbot(user_question, max_length=100)
    return jsonify({"answer": response[0]["generated_text"]})

if __name__ == "__main__":
    app.run(debug=True)
