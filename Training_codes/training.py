from ultralytics import YOLO
if __name__ == "__main__":
    model = YOLO("yolov8n.pt")

    model.train(
        data="yolo8_obj/data.yaml", 
        epochs=100, 
        patience=15, 
        batch=32, 
        save_period=5, 
        device=0, 
        project="yolov8_obj_models", 
        name="1st_train")

    model.val()

    model.export(format="tflite", half=True)