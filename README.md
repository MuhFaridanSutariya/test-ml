# Image Classification API

This FastAPI application provides an image classification service. It accepts an image file as input and predicts the class of the image based on a CNN Model (`trained_model.h5`).

## API Documentation

### Base URL
The API is accessible at `http://127.0.0.1:8000`.

### Endpoints

#### `POST /predict/`

This endpoint accepts an image file and returns the predicted class label.

##### Request
- **Method**: `POST`
- **URL**: `/predict/`
- **Body**: `multipart/form-data`
- **File Parameter**: 
    - `file`: An image file (PNG, JPG, etc.) to be classified.
    - The image is automatically resized to `256x256` and processed by the model.
  
##### Example Request

To test the API, you can use the `curl` command or any HTTP client (e.g., Postman, Insomnia):

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@path_to_your_image.jpg'
