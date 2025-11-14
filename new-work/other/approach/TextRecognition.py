from paddleocr import TextRecognition
model = TextRecognition(model_name="PP-OCRv5_server_rec")
output = model.predict(input="approach/img28.jpg", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="approach/output/TextRecognition.jpg")
    res.save_to_json(save_path="approach/output/TextRecognition_res.json")