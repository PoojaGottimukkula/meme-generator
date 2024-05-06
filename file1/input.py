#Image as input and meme as output
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import numpy as np
import cv2
import random

# Load the Motion Detector model
model = tf.keras.models.load_model('emotiondetector.h5')

# Define emotions
emotions = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# Assuming we have a list of dialogues for each emotion
emotions_dialogues = {
    "Angry": ["I am really mad right now!", "Why are you so annoying?"],
    "Happy": ["I'm so happy!", "Life is beautiful."],
    "Sad": ["I feel so down.", "Everything seems gloomy."],
    "Surprise": ["Wow, I didn't expect that!", "This is surprising!"],
    "Neutral": ["I'm feeling neutral.", "Nothing special."],
    "Fear":["I cant walk into a shop anywhere where i do not feel uncomfortable.","i hate it when i feel fearful for absolutely no reason"],
    "Disgust":["That was the most repulsive thing I've ever seen.","I'm sickened by their lack of hygiene."]
}

# Define a function to preprocess the image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (48, 48))
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

# Define a function to predict facial expression
def predict_expression(image):
    prediction = model.predict(image)
    expression_class = np.argmax(prediction)
    return emotions[expression_class]

# Define a function to generate dialogue/text based on expression
def generate_dialogue(expression):
    return random.choice(emotions_dialogues[expression])

# Define a function to overlay text on the image
def overlay_text(image, text):
    cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return image

# Create a Tkinter window
root = tk.Tk()
root.withdraw()  # Hide the root window

# Ask the user to select an image file
file_path = filedialog.askopenfilename()

if file_path:
    # Load the image
    image = cv2.imread(file_path)

    # Preprocess the image and predict expression
    try:
        preprocessed_image = preprocess_image(image)
        expression = predict_expression(preprocessed_image)
        
        # Generate dialogue based on expression
        dialogue = generate_dialogue(expression)
        
        # Overlay text on the image
        final_image = overlay_text(image.copy(), dialogue)
        
        # Display or save the final image with overlaid text
        cv2.imshow("Final Image", final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print("Error:", e)