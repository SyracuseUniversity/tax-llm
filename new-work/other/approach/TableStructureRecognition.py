from paddleocr import TableStructureRecognition
model = TableStructureRecognition(model_name="SLANet")
output = model.predict(input="approach/img28.jpg", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_json("approach/output/TableStructureRecognition_res.json")