# Flag Classification with YOLOv8

This project shows how to perform flag classification using YOLOv8. In this repository, we provide a pre-trained YOLOv8 model trained on a dataset of flags from different countries and then use the geotastic website to detect the flag and point the country on the world map.

The model was trained to classify 231 countries based on their flags.

<img src="examples\example1.gif" alt="GIF" width="700"/>

<img src="examples\example2.gif" alt="GIF" width="700"/>

<img src="examples\example4.gif" alt="GIF" width="700"/>

<img src="examples\example5.gif" alt="GIF" width="700"/>

## Requirements

Before running the code, ensure you have the following dependencies installed:
- Python 3.10
- OpenCV
- NumPy
- ultralytics (YOLOv8 implementation)
- PyQT5 (Application for capture screen)

You can install the required dependencies by running:
```bash
pip install -r requirements.txt
```

## YOLOv8 Training
1. Clone the repository:
git clone https://github.com/lilmarcin/geotastic_with_AI.git
cd geotastic_with_AI

2. For new training download the pre-trained YOLOv8 model weights or use `yolov8n-cls.pt`.
https://docs.ultralytics.com/tasks/classify/

3. Prepare dataset with specific organization [Ultralytics prepare dataset](https://docs.ultralytics.com/datasets/classify/)

```bash
datasets/
|
|-- train/
|   |-- Afghanistan/
|   |   |-- Afghanistan001.png
|   |   |-- Afghanistan002.png
|   |   |-- ...
|   |
|   |-- Albania/
|   |   |-- Albania001.png
|   |   |-- Albania002.png
|   |   |-- ...
|   |-- ...
|
|-- test/
|   |-- Afghanistan/
|   |   |-- Afghanistan001.png
|   |   |-- Afghanistan002.png
|   |   |-- ...
|   |
|   |-- Albania/
|   |   |-- Albania001.png
|   |   |-- Albania002.png
|   |   |-- ...
|   |-- ...
|
|-- val/ (optional)
|   |-- Afghanistan/
|   |   |-- Afghanistan001.png
|   |   |-- Afghanistan002.png
|   |   |-- ...
|   |
|   |-- Albania/
|   |   |-- Albania001.png
|   |   |-- Albania002.png
|   |   |-- ...
|   |-- ...
|
```

4. Run notebook to start training: `training_yolov8.ipynb`. Remember to change paths and set parametetrs for training.
```python
model = YOLO("yolov8n-cls.pt")
model.train(data="datasets/", epochs=50)
```

5. Evaluate the trained model using `testing_yolov8.ipynb`.
<img src="/runs/classify/val/val_batch0_labels.jpg" alt="Val batch0" width="700"/>
<img src="/runs/classify/val/val_batch1_labels.jpg" alt="Val batch1" width="700"/>
<img src="/runs/classify/val/val_batch2_labels.jpg" alt="Val batch2" width="700"/>



## Usage application on [Geotastic](https://geotastic.net/highscore-hunt)

1. Run the App:
```bash
python ScreenCaptureApp.py
```
<img src="examples/App1.png" alt="App" width="700"/>

2. Click `Start capture frame` and select the area where the flags will appear. Next press ENTER to confirm area.
<img src="examples/Test2.png" alt="1" width="700"/>

3. The App will output the predicted country based on the detected flag.
<img src="examples/Test3.png" alt="2" width="700"/>

4. Enter the image refresh interval (in seconds). Default is 1 second.
<img src="examples/Test4.png" alt="3" width="700"/>

5. Mark the predicted country on the map and click Finish Guess.
<img src="examples/Test5.png" alt="4" width="700"/>

## Examples

<img src="examples/Test3.png" alt="Angola" width="700"/>
<img src="examples/Test6.png" alt="Papua New Guinea" width="700"/>
<img src="examples/Test7.png" alt="Belize" width="700"/>


## License

This project is licensed under the [MIT License](LICENSE).