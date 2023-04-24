import os
import os.path as osp
import cv2 , time
from typing import Any , Dict , List
from segment_anything import SamPredictor , sam_model_registry # for prompt
from segment_anything import SamAutomaticMaskGenerator # for entire image


def write_masks_to_folder(masks : List[Dict[str , Any]] , path : str) ->None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i , mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(osp.join(path , filename) , mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return
        
# --- CONFIG START --- #
sam_vit_h = r"model_weights/sam_vit_h_4b8939.pth" # default , vit_h
sam_vit_b = r"model_weights/sam_vit_b_01ec64.pth" # vit_b
sam_vit_l = r"model_weights/sam_vit_l_0b3195.pth" # vit_l

checkpoint_path = sam_vit_h
image_path = r"../Data/NailsJpgfile/images/1_1-2.jpg"
prompts_flag = False
# --- CONFIG END --- #

assert osp.exists(image_path)
image = cv2.imread(image_path)
# cv2.imshow("img" , image)
# cv2.waitKey(0)


def main() -> None:
    print("Loading model...")
    # --- Regist SAM Model --- #
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    if prompts_flag :
        # get masks from a given prompt
        print("Get masks from a given prompt ... ")
        input_prompts = "" # prompt
        predictor = SamPredictor(sam)
        predictor.set_image(image)
    else :
        # generate masks for an entire image
        print("Generate masks for an entire image ... ")
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(image)
        print(masks)

def main_1():
    SUPPORT = ["png" , "jpg" , "jpeg"]
    output_dir = r"E:\OroChiLab\segment-anything-main\output"
    for dir in os.listdir(output_dir):
        image_name = dir
        org_image = f"E:\\OroChiLab\\Data\\NailsJpgfile\\images\\{image_name}.jpg"
        file_dir = f"E:\\OroChiLab\\segment-anything-main\\output\\{image_name}"
        save_dir = osp.join(file_dir , "with_mask")
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        for idx , file in enumerate(os.listdir(file_dir)):
            if file.split(".")[-1] not in SUPPORT:
                print(file.split(".")[-1])
                continue
            file_path = osp.join(file_dir , file)
            # print(file_path)
            alpha = 0.8
            meta = 1 - alpha
            gamma = 0
            srcImage = cv2.imread(org_image)
            mask = cv2.imread(file_path)
            result = cv2.addWeighted(srcImage , alpha , mask , meta , gamma)
            save_name = "with_mak" + osp.basename(file_path)
            # cv2.imshow("result" , result)
            # cv2.waitKey(0)
            save_path = osp.join(save_dir , save_name)
            print(save_path)
            cv2.imwrite(save_path , result)


if __name__ == "__main__":
    # main()
    main_1()