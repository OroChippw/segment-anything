import cv2
import os , os.path as osp
import warnings
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

from segment_anything import sam_model_registry , SamPredictor
from segment_anything.utils.onnx import SamOnnxModel

# ----------- CONFIG START ----------- #
MODEL_TYPE = "vit_b"

TORCH_MODEL_PATH = f"model_weights/sam_vit_b_01ec64.pth"
ONNX_MODEL_PATH = f"model_weights/sam_{MODEL_TYPE}.onnx"
ONNX_MODEL_QUANTIZED_PATH = f"model_weights/sam_quantized_{MODEL_TYPE}.onnx"
DEVICE = "cpu"
# ----------- CONFIG END ------------- #

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
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

class SAMOnnxRunner():
    def __init__(self , torch_model_path , onnx_model_path , model_type ,  device = "cuda") -> None:
        self.torch_path = torch_model_path
        self.onnx_path = onnx_model_path
        self.buildOnnx = False
        self.model_type = model_type
        self.device = device
        
        assert osp.exists(self.torch_path) , \
            f"{torch_model_path} is not exists"
        assert self.model_type is not None , \
            f"model_type should be set"
        
        if osp.exists(self.onnx_path) is None:
            print(f"{self.onnx_path} is not exists!")
            self.buildOnnx = True 
        
        # Init Torch Model and Onnx Model
        self.model = sam_model_registry[self.model_type](checkpoint=self.torch_path)

    # --------------------------------- #       
    def _export_onnx_model(self , onnx_save_path , use_quantize = False):
        sam_onnx = SamOnnxModel(self.model , return_single_mask=True)
        
        dynamic_axes = {
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        } 
        
        embed_dim = self.model.prompt_encoder.embed_dim
        embed_size = self.model.prompt_encoder.image_embedding_size
        mask_input_size = [4 * x for x in embed_size]
        
        dummy_inputs = {
           "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
            "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
            "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
            "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
            "has_mask_input": torch.tensor([1], dtype=torch.float),
            "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float), 
        }
        
        output_names = ["masks", "iou_predictions", "low_res_masks"]
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore" , category=UserWarning)
            with open(onnx_save_path , "wb") as f:
                torch.onnx.export(
                    sam_onnx,
                    tuple(dummy_inputs.values()),
                    f,
                    export_params=True,
                    verbose=True,
                    opset_version=17,
                    do_constant_folding=True,
                    input_names=list(dummy_inputs.keys()),
                    output_names=output_names,
                    dynamic_axes=dynamic_axes
                )
            print("Finish export onnx model...")
        if use_quantize:
            quantize_dynamic(
                model_input=ONNX_MODEL_PATH,
                model_output=ONNX_MODEL_QUANTIZED_PATH,
                optimize_model=True,
                per_channel=False,
                reduce_range=False,
                weight_type=QuantType.QUInt8
            )
            print("Finish quantize onnx model...")
            
    # --------------------------------- #       
    def _preprocess(self , srcImage_path , input_point , input_label):
        image = cv2.imread(srcImage_path)
        image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
        
        # Add a batch index, concatenate a padding point, and transform.
        onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
        onnx_coord = self.predictor.transform.apply_coords(onnx_coord , image.shape[:2]).astype(np.float32)
        
        # Create an empty mask input and an indicator for no mask.
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)
        
        # Package the inputs to run in the onnx model
        ort_inputs = {
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
        }
        
        return image , ort_inputs
    
    # --------------------------------- #       
    def _inference(self , image , ort_inputs):
        self.predictor.set_image(image)
        image_embedding = self.predictor.get_image_embedding().cpu().numpy()
        ort_inputs["image_embeddings"] = image_embedding
        
        # Build session 
        self.ort_session = ort.InferenceSession(self.onnx_path)
        
        # Predict a mask and threshold it.
        masks, _, low_res_logits = self.ort_session.run(None, ort_inputs)
        masks = masks > self.predictor.model.mask_threshold
        
        return masks
    
    # --------------------------------- #       
    def _postprocess(self):
        print("Finsh inference!")
        pass
    
    # --------------------------------- #       
    def _inference_img(self , image_path , point , label):
        if self.buildOnnx:
            print("buildONNX...")
            self._export_onnx_model(self.onnx_path)
        self.model.to(self.device)
        self.predictor = SamPredictor(self.model)
        
        img , ort_inputs= self._preprocess(image_path , point , label)
        masks = self._inference(img , ort_inputs)
        self._postprocess()
        return masks
        
        
    
def main():
    # Get Input Information
    input_point = np.array([[500, 375]])
    input_label = np.array([1])
    image_path = r"notebooks//images//truck.jpg"
    assert osp.exists(image_path) , \
        f"{image_path} is not exists!"
        
    # Build SAMOnnxRunner
    sam_onnx_runner = SAMOnnxRunner(TORCH_MODEL_PATH  , ONNX_MODEL_PATH , MODEL_TYPE , DEVICE)
    
    # Export vit onnx
    # sam_onnx_runner._export_onnx_model(onnx_save_path = ONNX_MODEL_PATH)
    
    # Inference by onnx
    masks = sam_onnx_runner._inference_img(image_path , input_point , input_label)
    
    print(masks.shape)
    
    plt.figure(figsize=(10,10))
    plt.imshow(cv2.imread(image_path))
    show_mask(masks, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.show() 
    
if __name__ == "__main__":
    main()

