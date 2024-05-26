import os
import cv2
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from src.paligemma import paligemma_parse
from src.paligemma.utility import slice_generator

class Paligemma:
    def __init__(self, model_id: str, access_token: str = os.getenv('HF_TOKEN')):
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, token=access_token)
        self.processor = AutoProcessor.from_pretrained(model_id, token=access_token)
    
    def run_inference(self, pil_image, prompt):
        width, height = pil_image.size
        inputs = self.processor(prompt, pil_image, return_tensors="pt")
        output = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
        decoded_output = self.processor.decode(output[0], skip_special_tokens=True)[len(prompt):].lstrip("\n")
        objs = paligemma_parse.extract_objs(decoded_output, width, height, unique_labels=True)
        return objs

    def run_sliced_inference(self, 
                             image, 
                             prompt, 
                             horizontal_stride=448,  
                             vertical_stride=448, 
                             max_det_h=200, 
                             max_det_w=200):
        
        slice_gen = slice_generator(
                image,
                horizontal_stride=horizontal_stride,
                vertical_stride=vertical_stride,
            )
        
        det_boxes = []
        for slice_crop, v_start, h_start in slice_gen:
            infer_image = Image.fromarray(slice_crop)
            result = self.run_inference(infer_image, prompt)
            if len(result) > 0:
                if 'xyxy' in result[0]:
                    x1,y1,x2,y2 = result[0]['xyxy']
                    height = int(y2-y1)
                    width = int(x2-x1)
                    if height > max_det_h or width > max_det_w:
                        continue
                    new_bbox = [x1+h_start, y1+v_start, x2+h_start, y2+v_start]
                    det_boxes.append(new_bbox) 
                    
        return det_boxes