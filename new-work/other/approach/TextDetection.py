from paddleocr import TextDetection
model = TextDetection(model_name="PP-OCRv5_server_det")
output = model.predict("approach/img28.jpg", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="approach/output/")
    res.save_to_json(save_path="approach/output/TextDetection_res.json")