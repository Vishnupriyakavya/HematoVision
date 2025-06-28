## HematoVision: Blood Cell Classification

This project is a web-based application for classifying blood cells from images. It uses a deep learning model built with TensorFlow and Keras, integrated into a Flask web application. Users can upload an image of a blood cell, and the application will predict its type.

##  Features

-   **Image-based Classification**: Predicts blood cell type from an uploaded image.
-   **Deep Learning Model**: Utilizes a pre-trained MobileNetV2 model for transfer learning, achieving accurate predictions.
-   **Web Interface**: A user-friendly interface for uploading images and viewing results.
-   **Four Cell Types**: Classifies four major types of white blood cells:
    -   Eosinophil
    -   Lymphocyte
    -   Monocyte
    -   Neutrophil

## Technologies Used

-   **Backend**: Python, Flask
-   **Machine Learning**: TensorFlow, Keras, Scikit-learn
-   **Frontend**: HTML, CSS
-   **Libraries**: NumPy

##  Project Structure

```
HematoVision/
├── static/
│   ├── css/
│   │   └── style.css
│   └── uploads/
├── templates/
│   ├── home.html
│   └── result.html
├── app.py
├── train_model.py
├── Blood Cell.h5
├── requirements.txt
└── README.md
```



## Usage

There are two main parts to this project: training the model and running the web application.

### 1. Training the Model

The `Blood Cell.h5` model file is already included. However, if you want to retrain the model with new data, you can run the training script.

-   **Place your dataset** in the `dataset2-master` directory in the project root. The dataset should have `TRAIN` and `TEST` subdirectories, each containing folders for the different cell types.
-   **Run the training script:**
    ```bash
    python train_model.py
    ```
    This will generate a new `Blood Cell.h5` file in the root directory.

### 2. Running the Web Application

-   **Start the Flask server:**
    ```bash
    python app.py
    ```

-   **Open your web browser** and navigate to:
    [http://127.0.0.1:5000](http://127.0.0.1:5000)

-   **Upload an image** of a blood cell and click "Predict" to see the result. 
