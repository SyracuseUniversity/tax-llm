from paddleocr import LayoutDetection

model = LayoutDetection(model_name="PP-DocLayout_plus-L")
output = model.predict("approach/page_29.png", batch_size=1, layout_nms=True)
for res in output:
    res.print()
    res.save_to_img(save_path="approach/output")
    res.save_to_json(save_path="approach/output/LayoutDetection_res.json")