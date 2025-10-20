from typing import Optional, Tuple, Union, List, Dict, Iterable
from PIL import Image
import numpy as np
import torch

def add_image_tokens_to_prompt(prefix_prompt, bos_token, img_seq_len, image_token):
    
    return f"{image_token * img_seq_len}{bos_token} {prefix_prompt}\n"
    

def resize(
    image: Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None
    ) -> np.ndarray:
    height, width = size
    resized_image = image.resize((width, height), resample=resample)
    return resized_image

def rescale(
    image: np.ndarray,
    scale_factor: float,
    dtype:np.dtype = np.float32
    ) -> np.ndarray:
    rescaled_image = image * scale_factor
    rescaled_image = rescaled_image.astype(dtype)
    return  rescaled_image

def normalize(
    image: np.ndarray,  
    mean: Union[float,Iterable[float]],
    std: Union[float,Iterable[float]]
    ) -> np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image


def process_image(
    imsges : List[Image.Image],
    size: Dict[str, int]=None,
    resample: Image.Resampling=None,
    resample_factor: float =None,
    image_mean: Optional[Union[List[float]]]=None,
    image_std: Optional[Union[List[float]]]=None
    )->List[np.ndarray]:
    height, width = size[0], size[1]
    images = [
        resize(image=image, size=(height,width), resample=resample) for image in imsges
    ]
    images = [np.array(image) for image in images]
    images = [np.rescale(image) for image in images]
    images = [np.normalize(image) for image in images]
    images = [image.transpose(2,0,1) for image in images] #HWC to CHW
    return images

IMAGENET_STANDARD_MEAN = [0.5,0.5,0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

class PaliGemmaProcessor:
    IMAGE_TOKEN =   "<image>"
    
    def __init__(self,tokenizer, num_img_tokens:int, img_size:int):
        super().__init__()
        self.tokenizer = tokenizer
        self.num_img_tokens = num_img_tokens
        self.img_size = img_size
        
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [f"<loc{i:04d}>" for i in range(1024)]
        EXTRA_TOKENS+= [f"<seg{i:03d}>" for i in range(1024)]
        tokenizer.add_special_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
        
        self.tokenizer = tokenizer
        
    def __call__(self,text:List[str], images:List[Image.Image], padding:str = 'longest',trunction:bool=True)->dict:
        assert len(text) == 1 and len(images)==1, f"Recieved {len(text)} texts and {len(images)} "
        
        pixel_values = self.process_images(
            images,
            size=(self.img_size, self.img_size)
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1/255.0
            image_mean =     IMAGENET_STANDARD_MEAN,
            image_std = IMAGENET_STANDARD_STD
        )
        
        pixel_values = np.stack(pixel_values)#list of arrays to array
        pixel_values = torch.tensor(pixel_values)
        
        #Prepend image tokens to text
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_token=self.IMAGE_TOKEN,
                image_seq_len=self.img_seq_len
            )
            for prompt in text
        ]
        
        #REturn the input ids and attention mask
        inputs = self.tokenizer(
            input_strings,
            padding=padding,
            truncation=trunction,
            return_tensors="pt"
        )
        
        return_data = {"pixel_values": pixel_values, **inputs}
        return return_data
        