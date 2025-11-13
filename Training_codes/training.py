from ultralytics import YOLO
if __name__ == "__main__":
    model = YOLO("yolov8n.pt")

    model.train(
        data="Yolov8.v2i.yolov8/data.yaml", 
        epochs=100, 
        patience=15, 
        batch=16, 
        save_period=5, 
        device=0, 
        project="yolov8_obj_models_v2", 
        name="1st_train")

    model.val()

    model.export(format="tflite", half=True)