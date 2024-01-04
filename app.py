import gradio as gr
from transformers import pipeline
from PIL import Image, ImageOps
import time
from rembg import remove

# Initialize Segmentation Pipeline
segformer_b2_clothes_pipe = pipeline("image-segmentation", model="mattmdjaga/segformer_b2_clothes")

def segformer_b2_clothes(img):
    result = segformer_b2_clothes_pipe(img)
    mask = result[0]['mask'].convert('L')
    mask = ImageOps.invert(mask)
    img.putalpha(mask)
    return img

def rembg_remove(img):
    return remove(img)

def remove_background(img):
    # segformer_b2_clothes
    start = time.time()
    segformer_b2_clothes_result = segformer_b2_clothes(img)
    end = time.time()
    segformer_b2_clothes_text = """[mattmdjaga/segformer_b2_clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes) \n""" + str(end-start) + """ seconds"""

    #rembg
    start = time.time()
    rembg_result = rembg_remove(img)
    end = time.time()
    rembg_text = "[rembg](https://huggingface.co/spaces/openskyml/remove-background-on-image) \n" + str(end-start) + " seconds"
    
    return segformer_b2_clothes_text, segformer_b2_clothes_result, rembg_text, rembg_result 

iface = gr.Interface(fn=remove_background, 
                     inputs=gr.Image(type='pil'), 
                     outputs=[gr.Markdown(),
                              gr.Image(label='segformer_b2_clothes', type='pil'),
                              gr.Markdown(),
                              gr.Image(label='rembg', type='pil')])
iface.launch()