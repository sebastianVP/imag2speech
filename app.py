from transformers import TrOCRProcessor,VisionEncoderDecoderModel
import gradio as gr
import torch

#processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-spanish")
#model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-spanish")

processor = TrOCRProcessor.from_pretrained("qantev/trocr-base-spanish")
model = VisionEncoderDecoderModel.from_pretrained("qantev/trocr-base-spanish")

type = "cuda" if torch.cuda.is_available() else "cpu"
model.to(type)

def process(image):
    with torch.no_grad():
        pixels = processor(image, return_tensors="pt").pixel_values.to(type)
        ids = model.generate(pixels)
        text= processor.batch_decode(ids,skip_special_tokens=True)[0]

    return text

iface= gr.Interface(
    fn=process,
    inputs = gr.Image(type="pil",label="Imagen"),
    outputs=gr.Textbox(label="Texto Encontrado")     
    )

iface.launch(debug=True)


