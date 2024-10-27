from flask import Flask, request, jsonify, render_template
import re
import random
import cv2
import base64
import numpy as np
from deepface import DeepFace
import requests
from io import BytesIO
from PIL import Image

# Set up Flask app
app = Flask(__name__)

# Set your API endpoint and key for the AI model here (Placeholder for Gemini AI)
GEMINI_API_KEY = "your_gemini_api_key"  # Replace with actual Gemini API key when available
GEMINI_API_ENDPOINT = "https://gemini-ai.api.endpoint/generate"  # Hypothetical API endpoint

class Radha:
    def __init__(self):
        self.reflections = {
            "am": "are", "was": "were", "i": "you", "i'd": "you would", "i've": "you have",
            "i'll": "you will", "my": "your", "are": "am", "you've": "I have", "you'll": "I will",
            "your": "my", "yours": "mine", "you": "me", "me": "you"
        }

        self.emotion_keywords = {
            'happy': ['I\'m glad to see you happy!', 'Keep smiling!'],
            'sad': ['I\'m sorry to see you sad.', 'It\'s okay, things will get better.'],
            'angry': ['Why do you look angry?', 'Would you like to talk about it?'],
            'surprise': ['You seem surprised!', 'What’s surprising you?'],
            'fear': ['You seem scared. Is everything okay?', 'Let’s talk about what’s making you fearful.'],
            'neutral': ['You seem calm.', 'How are you really feeling today?']
        }

        self.patterns = [
            (r'I need (.*)', ['Why do you need {0}?', 'Would it really help you to get {0}?', 'Are you sure you need {0}?']),
            (r'Why don\'t you (.*)', ['Do you really think I don\'t {0}?', 'Perhaps eventually I will {0}.', 'Do you want me to {0}?'])
        ]

    def reflect(self, fragment):
        tokens = fragment.lower().split()
        for i, token in enumerate(tokens):
            if token in self.reflections:
                tokens[i] = self.reflections[token]
        return ' '.join(tokens)

    def detect_emotion_text(self, statement):
        for emotion, responses in self.emotion_keywords.items():
            if emotion in statement.lower():
                return emotion
        return None

    def analyze(self, statement):
        emotion = self.detect_emotion_text(statement)
        if emotion:
            return random.choice(self.emotion_keywords[emotion])

        for pattern, responses in self.patterns:
            match = re.match(pattern, statement.rstrip(".!"))
            if match:
                response = random.choice(responses)
                return response.format(*[self.reflect(g) for g in match.groups()])

        return "Tell me more."

    def get_gemini_response(self, statement, emotion=None):
        prompt = f"The user feels {emotion}. Respond appropriately to this statement: {statement}" if emotion else f"Respond to this statement: {statement}"

        try:
            # Hypothetical API call to Gemini AI
            response = requests.post(
                GEMINI_API_ENDPOINT,
                headers={"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"},
                json={"prompt": prompt, "max_tokens": 150, "temperature": 0.7}
            )

            if response.status_code == 200:
                gemini_output = response.json()
                return gemini_output.get("choices", [{}])[0].get("text", "Sorry, I didn't understand that.")
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Sorry, I couldn't process the request: {e}"

    def detect_emotion_camera(self, image_data):
        try:
            # Decode base64 image data and convert it to a numpy array
            image = Image.open(BytesIO(base64.b64decode(image_data.split(',')[1])))
            image_np = np.array(image)

            # Analyze emotions using DeepFace
            emotion_analysis = DeepFace.analyze(image_np, actions=['emotion'], enforce_detection=False)
            dominant_emotion = emotion_analysis['dominant_emotion']
            return dominant_emotion
        except Exception as e:
            return None


# Create an instance of the Radha bot
radha = Radha()


# Define the routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/text_input', methods=['POST'])
def text_input():
    user_input = request.json.get('message', '')
    response = radha.analyze(user_input)
    return jsonify({"response": response})


@app.route('/camera_input', methods=['POST'])
def camera_input():
    image_data = request.json.get('image', '')
    detected_emotion = radha.detect_emotion_camera(image_data)
    if detected_emotion:
        response = f"Detected emotion: {detected_emotion}"
    else:
        response = "Could not detect emotion. Please try again."
    return jsonify({"response": response})


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
