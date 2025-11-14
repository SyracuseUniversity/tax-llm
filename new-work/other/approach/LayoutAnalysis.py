from paddleocr import LayoutDetection

model = LayoutDetection(model_name="PP-DocLayoutV2")
output = model.predict("approach/img28.jpg", batch_size=1, layout_nms=True)
for res in output:
    res.print()
    res.save_to_img(save_path="approach/output")
    res.save_to_json(save_path="approach/output/LayoutAnalysis_res.json")