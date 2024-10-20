from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig, AutoModel
from PIL import Image
import requests
from datasets import load_dataset

#url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
dataset = iter(load_dataset("imagenet-1k", split="train", streaming=True))
image_0 = next(dataset)["image"]
image_1 = next(dataset)["image"]
images = [image_0, image_1]
print(image_0.mode)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
inputs = processor(images=images, return_tensors="pt")
print(inputs.keys())
print(inputs["pixel_values"].shape)

'''model_config = ViTConfig(patch_size=1)
model = AutoModel.from_config(model_config)
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)'''

ind = 0
while True:
    try:
        im1 = next(dataset)["image"]
        im2 = next(dataset)["image"]
        if im1.mode == "RGB" and im2.mode == "RGB":
            processor(images=[im1, im2], return_tensors="pt")
        ind += 1
        print(ind)
    except Exception as e:
        print(e)
        print(50*"*")
        print(im1)
        print(50*"*")
        print(im2)
        break