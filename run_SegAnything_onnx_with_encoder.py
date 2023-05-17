import cv2
import os , os.path as osp
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
import onnxruntime as ort

from segment_anything.utils.transforms import ResizeLongestSide

# ----------- CONFIG START ----------- #
SUPPORTED = ["png" , "jpg" , "jpeg"]
MODEL_TYPE = "vit_b"
# MODEL_LIST = [sam_vit_b_01ec64  sam_vit_l_0b3195 sam_vit_h_4b8939]
TORCH_MODEL_PATH = f"model_weights/sam_vit_b_01ec64.pth"

ENCODER_ONNX_MODEL_PATH = f"model_weights/withEncoder/{MODEL_TYPE}/encoder.onnx"

USE_SINGLEMASK = False
if USE_SINGLEMASK:
    DECODER_ONNX_MODEL_PATH = f"model_weights/sam_{MODEL_TYPE}_singlemask.onnx"
    
else:
    # DECODER_ONNX_MODEL_PATH = f"model_weights/sam_{MODEL_TYPE}.onnx"
    DECODER_ONNX_MODEL_PATH = f"model_weights/withEncoder/{MODEL_TYPE}/decoder.onnx"
    
    
USE_QUANTIZED = False
if USE_QUANTIZED:
    ENCODER_ONNX_MODEL_QUANTIZED_PATH = f"model_weights/withEncoder/{MODEL_TYPE}/encoder-quant.onnx"
    DECODER_ONNX_MODEL_QUANTIZED_PATH = f"model_weights/withEncoder/{MODEL_TYPE}/decoder-quant.onnx"
    ENCODER_ONNX_MODEL_PATH = ENCODER_ONNX_MODEL_QUANTIZED_PATH
    DECODER_ONNX_MODEL_PATH = DECODER_ONNX_MODEL_QUANTIZED_PATH
    
DEVICE = "cpu"
SHOW = False
time_total = 0.0

USE_BOX = True

# ----------- CONFIG END ------------- #

# ----------- SHOW RESULT FUNC START ------------- #
def show_mask(masks, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = masks.shape[-2:]
    if not USE_SINGLEMASK:
        for i, mask in enumerate(masks[0]):
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)
    else:
        mask_image = masks.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  
# ----------- SHOW RESULT FUNC END ------------- #

# ----------- SAVE MASK FUNC START ------------- #
def save_masks(masks , savepath , filename):
    h, w = masks.shape[-2:]
    index = 0
    if not USE_SINGLEMASK:
        for idx, mask in enumerate(masks[0]):
            mask_image = mask.reshape(h, w, 1) * 255
            filename_ = filename + f"_{idx}.png"
            cv2.imwrite(osp.join(savepath , filename_) , mask_image)
            print(f"Save as : {osp.join(savepath , filename_)}")
            index += 1
    else:
        mask_image = masks.reshape(h, w, 1) * 255
        cv2.imwrite(osp.join(savepath , filename + ".png") , mask_image)
        print(f"Save as : {osp.join(savepath , filename)}")

# ----------- SAVE MASK FUNC END ------------- #



class SAMOnnxRunner():
    def __init__(self , encoder_path , decoder_path , model_type ,  device = "cuda") -> None:
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.model_type = model_type
        self.device = device
        self.transform = None
        self.init_encoder = False
        self.embedding = None
        self.mask_threshold = 0.0
        
        
        assert osp.exists(self.encoder_path) , f"{encoder_path} is not exists"
        assert osp.exists(self.decoder_path) , f"{decoder_path} is not exists"
        assert self.model_type is not None , \
            f"model_type should be set"
    
        # Init Onnx InferenceSession
        self.encoder_session = ort.InferenceSession(self.encoder_path)
        self.decoder_session = ort.InferenceSession(self.decoder_path)
            
    # --------------------------------- #
    def _preprocess_image(self , srcImage):
        image = cv2.cvtColor(srcImage , cv2.COLOR_BGR2RGB)
        return image
    
    def _preprocess_encoder(self , image):
        try:
            # => Preprocess for Encoder
            # Meta AI training encoder with a resolution of 1024*1024
            encoder_input_size = 1024
            self.transform = ResizeLongestSide(encoder_input_size)
            input_img = self.transform.apply_image(image)
            input_img_torch = torch.as_tensor(input_img , device=self.device)
            input_img_torch = input_img_torch.permute(2,0,1).contiguous()[None , : , : ,:]
            pixel_mean = torch.Tensor([123.675,116.28,103.53]).view(-1,1,1)
            pixel_std = torch.Tensor([58.395,57.12,57.375]).view(-1,1,1)
            x = (input_img_torch - pixel_mean) / pixel_std
            h , w = x.shape[-2:]
            padh = encoder_input_size - h
            padw = encoder_input_size - w
            # F.pad (左边填充数， 右边填充数， 上边填充数， 下边填充数)
            # 前两个参数对最后一个维度有效，后两个参数对倒数第二维有效
            x = F.pad(x , (0 , padw , 0 , padh)) 
            x = x.numpy()
            
        except:
            raise Exception
        
        encoder_inputs = {
            "x" : x,
        }
        
        return encoder_inputs
        
    def _preprocess_decoder(self , image , input_point , input_label , input_box = None):
        
        # => Preprocess for Decoder
        # Add a batch index, concatenate a padding point, and transform.
        temp = input_point
       
        temp = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)
       
        
        if USE_BOX:
            onnx_box_coords = input_box.reshape(2,2)
            onnx_box_labels = np.array([2,3])
            print("onnx_box_coords : " , onnx_box_coords)
            print("onnx_box_coords shape : " ,onnx_box_coords.shape)
            
            print("onnx_box_labels : " , onnx_box_labels)
            print("onnx_box_labels shape : " ,onnx_box_labels.shape)
            onnx_coord = np.concatenate([input_point, onnx_box_coords], axis=0)[None, :, :]
            onnx_label = np.concatenate([input_label, onnx_box_labels], axis=0)[None, :].astype(np.float32)
            print("onnx_coord shape : " ,onnx_coord.shape)
            print("onnx_label shape : " ,onnx_label.shape)
            
            
        else :
            onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
            onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
        
        onnx_coord = self.transform.apply_coords(onnx_coord , image.shape[:2]).astype(np.float32)
        print("apply coords : " , onnx_coord)
        
        # Create an empty mask input and an indicator for no mask.
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)
        orig_im_size = np.array(image.shape[:2], dtype=np.float32)
        
        # Package the inputs to run in the onnx model
        decoder_inputs = {
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
        }
        
        return image , decoder_inputs
    
    # --------------------------------- #
    def _inference_encoder(self , encoder_input):
        time_start = time.time()
        encoder_output = self.encoder_session.run(None , encoder_input)
        time_end = time.time()
        print(f"Encoder build image embedding cost time : {time_end - time_start}" , "s")
        return encoder_output
           
    def _inference_decoder(self , ort_inputs):
       
        ort_inputs["image_embeddings"] = self.image_embedding
        
        time_start = time.time()
        # Predict a mask and threshold it.
        masks , iou_predictions , low_res_logits = self.decoder_session.run(None, ort_inputs)
        
        
        time_end = time.time()
        time_cost = time_end - time_start
        print(f"Decoder generate mask cost time : {time_cost}" , "s")
       
        masks = masks > self.mask_threshold
        
        global time_total
        time_total += time_cost
        
        return masks , iou_predictions , low_res_logits
    
    # --------------------------------- #       
    def _postprocess(self):
        print("Finsh inference!")
        pass
    
    # --------------------------------- #       
    def _inference_img(self , srcImage , point , label , box = None):
        image = self._preprocess_image(srcImage)
        
        if not self.init_encoder:
            encoder_inputs = self._preprocess_encoder(image)
            encoder_outputs = self._inference_encoder(encoder_inputs)
            self.image_embedding = encoder_outputs[0]
            self.init_encoder = True
        
        img , ort_inputs= self._preprocess_decoder(image , point , label , box)
        masks , iou_predictions , low_res_logits = self._inference_decoder(ort_inputs)
        self._postprocess()
        return masks , iou_predictions , low_res_logits
        
        
def main():
    image_dir = r"E:\OroChiLab\Data\NailsJpgfile\images\test"
    if USE_SINGLEMASK:
        save_dir = r"data//result//singlemask"
    else:
        save_dir = r"data//result//multimask"
    
    save_dir = r"data//result//encoder_test"
    
    # Get Input Information
    input_point = np.array([[1156, 550]])
    input_label = np.array([1])
    if USE_BOX :
        input_box = np.array([773,187,1465,896]) # x1 ,y1 , x2 , y2左上角点及右下角点
    
     # Build SAMOnnxRunner
    sam_onnx_runner = SAMOnnxRunner(ENCODER_ONNX_MODEL_PATH , DECODER_ONNX_MODEL_PATH , MODEL_TYPE , DEVICE)
    
    file_counter = 0
    
    for image_file in os.listdir(image_dir):
        file_path = osp.join(image_dir , image_file)
        print(f"Processing : {file_path}")
        filename = osp.basename(file_path).split(".")[0]
        if file_path.split(".")[-1] not in SUPPORTED:
            print(f"Unsupport {file_path}")
            continue
        srcImg = cv2.imread(file_path)

        # Inference by onnx
        if USE_BOX:
            masks , iou_predictions , low_res_logits = sam_onnx_runner._inference_img(srcImg , input_point , input_label , input_box)
        else :
            masks , iou_predictions , low_res_logits = sam_onnx_runner._inference_img(srcImg , input_point , input_label)
            
        file_counter += 1
        print("masks shape : " , masks.shape)
        print(f"iou_predictions : {iou_predictions} , shape : {iou_predictions.shape}")

        save_path = osp.join(save_dir , filename)
        if not osp.exists(save_path):
            os.mkdir(save_path)
        save_masks(masks , save_path , filename)
        # save_masks(low_res_logits , save_path , filename)
        
        if SHOW:
            plt.figure(figsize=(10,10))
            srcImg = cv2.cvtColor(srcImg , cv2.COLOR_BGR2RGB)
            plt.imshow(srcImg)
            show_mask(masks, plt.gca())
            show_points(input_point, input_label, plt.gca())
            plt.axis('off')
            plt.show() 
    
    print(f"file_counter : " , file_counter)
    print(f"Mean cost time : {time_total / file_counter}" , "s")
    
if __name__ == "__main__":
    main()

