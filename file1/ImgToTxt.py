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
    4: "Sadness",
    5: "Surprise",
    6: "Neutral"
}

# Specify the path to the text file
dataset_path = "dataset2/train.txt"

# Initialize an empty dictionary to store dialogues by emotion
emotions_dialogues = {}

# Read the content of the text file
with open(dataset_path, 'r') as file:
    lines = file.readlines()

# Parse each line to extract dialogues and emotions
for line in lines:
    dialogue, emotion = line.strip().split(';')
    
    # Check if the emotion is already present in the dictionary
    if emotion in emotions_dialogues:
        # If yes, append the dialogue to the existing list
        emotions_dialogues[emotion].append(dialogue)
    else:
        # If no, create a new list with the dialogue
        emotions_dialogues[emotion] = [dialogue]

# Define a function to preprocess the image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (48, 48))
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize pixel values
    return image

# Define a function to predict facial expression
def predict_expression(image):
    prediction = model.predict(image)
    expression_class = np.argmax(prediction)
    return emotions[expression_class]

# Define a function to generate dialogue/text based on expression
def generate_dialogue(expression):
    # Convert expression to lowercase for case-insensitive match
    expression = expression.lower()
    
    if expression in emotions_dialogues:
        return random.choice(emotions_dialogues[expression])
    else:
        # Fallback dialogue if the expression is not found in the dataset
        return "unrecognisable!!!!!!!!!!!!!!"

# Define a function to overlay text on the image
def overlay_text(image, text):
    cv2.putText(image, text+"---"+expression, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
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
        final_image =overlay_text(image.copy(), dialogue)
        
        #final_image = overlay_text(image.copy(), f"Emotion: {expression}\nDialogue: {dialogue}")
        
        # Display or save the final image with overlaid text
        cv2.imshow("Final Image", final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print("Error:", e)
