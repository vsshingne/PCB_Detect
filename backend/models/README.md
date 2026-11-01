# Custom model location

Copy the trained Ultralytics YOLOv8 weights from Google Colab to this directory as:

`pcb_defect_best.pt`

The backend loads this file at startup. It does not download or substitute pretrained weights. Until the custom model is present and its six class names match the configured order, health reports prediction unavailable and `/api/predict` returns HTTP 503.
