# inference.sh
python scripts/amg.py --checkpoint E:\\OroChiLab\\segment-anything-main\\model_weights\\sam_vit_h_4b8939.pth --model-type vit_h --input E:\\OroChiLab\\Data\\Stone\\test --output E:\\OroChiLab\\segment-anything-main\\output_stone

# export model shell
python scripts/export_onnx_model.py --checkpoint E:\\OroChiLab\\segment-anything-main\\model_weights\\sam_vit_h_4b8939.pth --model-type vit_h --output E:\\OroChiLab\\segment-anything-main\\onnx_models