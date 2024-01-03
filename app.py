import gradio as gr
from transformers import pipeline
from PIL import Image, ImageOps

# Initialize Segmentation Pipeline
segformer_b2_clothes_pipe = pipeline("image-segmentation", model="mattmdjaga/segformer_b2_clothes")

def segformer_b2_clothes(img):
    result = segformer_b2_clothes_pipe(img)
    mask = result[0]['mask'].convert('L')
    mask = ImageOps.invert(mask)
    img.putalpha(mask)
    return img

def remove_background(img):
    segformer_b2_clothes_result = segformer_b2_clothes(img)
    
    return segformer_b2_clothes_result

iface = gr.Interface(fn=remove_background, 
                     inputs=gr.Image(type='pil'), 
                     outputs=gr.Image(label='segformer_b2_clothes', type='pil'))
iface.launch()