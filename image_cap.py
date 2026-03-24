import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Initialize BLIP components - processor to tokenize input and model to generate caption
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Getting the image ready for the BLIP to caption
img_url = "lion.jpg"
# convert image to RGB format
image = Image.open(img_url).convert('RGB')

# Set the input for the processor
text = "the image of "
# Put in the image, the text and the pytorch ('pt') tensors
inputs = processor(images=image, text=text, return_tensors="pt")

# ** to unpack dictionaries and pass dict items as keyword args to the func
outputs = model.generate(**inputs, max_length=50) # 50 tokens in length

# Decode the caption generate by BLIP into readable text format
# skip_special_tokens = True to ignore special tokens like [SEP], [CLS], [PAD], [UNK]
caption = processor.decode(outputs[0], skip_special_tokens=True)
print(caption)