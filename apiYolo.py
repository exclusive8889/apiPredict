from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request
import os
import my_yolov6
import cv2

# Khởi tạo Flask Server Backend
app = Flask(__name__)
CORS(app)

# Apply Flask CORS
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = "static"
app.config['PREVIEW'] = "preview"
yolov6_model = my_yolov6.my_yolov6(
    "weights/best-train.pt", "cpu", "data/coco.yaml", 640, False)


@app.route('/predict', methods=['POST'])
def predict_yolov6():
    image = request.files.getlist("file")
    path_pred = []
    if image:
        for f in image:

            path_to_save = os.path.join(
                app.config['UPLOAD_FOLDER'], f.filename)
            f.save(path_to_save)
            # print("Save = ", path_to_save)

            frame = cv2.imread(path_to_save)
            # # Nhận diên qua model Yolov6
            frame, no_object = yolov6_model.infer(frame)

            pre = {
                "img": path_to_save,
                "number": no_object
            }

            if no_object > 0:
                cv2.imwrite(path_to_save, frame)
            del frame
            # Trả về đường dẫn tới file ảnh đã bounding box
            path_pred.append(pre)

        return {"data": path_pred}  # http://server.com/static/path_to_save

    return 'Upload file to detect'


@app.route('/preview', methods=['POST'])
def preview():
    image = request.files.getlist("file")
    path_pred = []
    if image:
        for f in image:

            path_to_save = os.path.join(
                app.config['PREVIEW'], f.filename)
            f.save(path_to_save)
            # print("Save = ", path_to_save)

            # frame = cv2.imread(path_to_save)
            # # Nhận diên qua model Yolov6
            # frame, no_object = yolov6_model.infer(frame)

            pre = {
                "img": path_to_save,
                # "number": no_object
            }

            # if no_object > 0:
            # cv2.imwrite(path_to_save, frame)
            # del frame
            # Trả về đường dẫn tới file ảnh đã bounding box
            path_pred.append(pre)

        return {"data": path_pred}  # http://server.com/static/path_to_save

    return 'Upload file to detect'


# Start Backend
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='6868')
