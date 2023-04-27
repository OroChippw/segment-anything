# inference.sh
# vit_h
python scripts/amg.py --checkpoint E:\\OroChiLab\\segment-anything-main\\model_weights\\sam_vit_h_4b8939.pth --model-type vit_h --input E:\\OroChiLab\\Data\\NailsJpgfile\\images --output E:\\OroChiLab\\segment-anything-main\\data\\output_nails

# vit_l
python scripts/amg.py --checkpoint E:\\OroChiLab\\segment-anything-main\\model_weights\\sam_vit_l_0b3195.pth --model-type vit_l --input E:\\OroChiLab\\Data\\NailsJpgfile\\images --output E:\\OroChiLab\\segment-anything-main\\data\\output_nails

# vit_b
python scripts/amg.py --checkpoint E:\\OroChiLab\\segment-anything-main\\model_weights\\sam_vit_b_01ec64.pth --model-type vit_b --input E:\\OroChiLab\\Data\\NailsJpgfile\\images --output E:\\OroChiLab\\segment-anything-main\\data\\output_nails


# export model shell
python scripts/export_onnx_model.py --checkpoint E:\\OroChiLab\\segment-anything-main\\model_weights\\sam_vit_h_4b8939.pth --model-type vit_h --output onnx_models\\sam_vit_h.onnx --return-single-mask