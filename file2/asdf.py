import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from nltk.sentiment import SentimentIntensityAnalyzer

# Step 1: Detect Emotion in Text
def detect_emotion(text):
    # Initialize SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    
    # Get sentiment scores
    scores = sid.polarity_scores(text)
    
    # Determine emotion based on sentiment scores
    if scores['compound'] >= 0.05:
        return 'joy'
    elif scores['compound'] <= -0.05:
        return 'sadness'
    elif scores['compound'] >= 0.3:
        return 'love'
    elif scores['compound'] <= -0.3:
        return 'fear'
    elif scores['compound'] <= -0.1:
        return 'angry'
    else:
        return 'neutral'

# Step 2: Search for Related Images using Google Custom Search API
def search_for_images(query, api_key, cx):
    # Construct the URL for the Google Custom Search API request
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={query}&searchType=image"
    
    # Send the request to the Google Custom Search API
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        
        # Extract image URLs from the response
        image_urls = []
        if 'items' in data:
            for item in data['items']:
                if 'link' in item:
                    image_urls.append(item['link'])
        
        return image_urls
    else:
        # Request was unsuccessful
        print(f"Failed to fetch images. Status code: {response.status_code}")
        return None

# Step 3: Overlay Text on the Image
def overlay_text_on_image(image_url, text):
    # Check if the image URL is empty
    if not image_url:
        print("No image available for the detected emotion.")
        return

    # Download the image from the URL
    response = requests.get(image_url)
    if response.status_code != 200:
        print(f"Failed to download image from URL: {image_url}")
        return

    # Open the image using PIL
    image = Image.open(BytesIO(response.content))

    # Add text overlay
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((10, 10), text, fill="white", font=font)

    # Display the final image
    image.show()

# Main Function
def main(input_text):
    # Step 1: Detect Emotion in Text
    detected_emotion = detect_emotion(input_text)
    print("emotion :", detected_emotion)
    # Google Custom Search API Key and Custom Search Engine ID
    api_key = "AIzaSyCEU_HUGZ4MhycnhhOB0Ucy-K0HBcModEg"
    cx = "858148054fa214df9"

    # Step 2: Search for Related Images
    image_urls = search_for_images(detected_emotion, api_key, cx)

    # Step 3: Overlay Text on the Image
    if image_urls:
        for image_url in image_urls:
            overlay_text_on_image(image_url, input_text)
    else:
        print("No images found for the detected emotion.")

# Entry Point
if __name__ == "__main__":
    input_text = input("Enter the text: ")
    main(input_text)
