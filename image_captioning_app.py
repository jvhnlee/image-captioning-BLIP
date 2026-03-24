import requests
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
import gradio

# Initialize BLIP components - processor to tokenize input and model to generate caption
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(input_image: np.ndarray):
    # Convert the np img to PIL Image then convert to RGB format
    raw_image = Image.fromarray(input_image).convert('RGB')

    # Set the input for the processor
    text = "the image of "
    # Put in the image, the text and the pytorch ('pt') tensors
    inputs = processor(images=raw_image, text=text, return_tensors="pt")

    # ** to unpack dictionaries and pass dict items as keyword args to the func
    outputs = model.generate(**inputs, max_length=50) # 50 tokens in length

    # Decode the caption generate by BLIP into readable text format
    # skip_special_tokens = True to ignore special tokens like [SEP], [CLS], [PAD], [UNK]
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

interface = gradio.Interface(
    fn=caption_image,
    inputs=gradio.Image(),
    outputs="text",
    title="Image Captioning",
    description="Web app for generating image captions using trained model"
)

interface.launch()