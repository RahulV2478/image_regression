# Homekynd-Convergent
Description: This is an image regression CNN Pytorch model that predicts the bounding box location of a couch in an unfurnished room.


Instructions
1. Clone this repository
2. Create and activate a virtual environment:
- For Mac:
   - **python -m venv [ENV_NAME]**
   - **source [ENV_NAME]/bin/activate**
3. Install required dependencies:
- **pip install -r requirements.txt**
4. Checkout to the image_regression branch (**git checkout image_regression**) and enter the image-regression folder (**cd image-regression**)
5. The example_dataset folder contains train and test data. The data is unfurnished rooms with the coordinates of the couch in the corresponding furnished image.
6. Run **python network.py**
- This will begin training the CNN
- First, a few train images will be displayed. Close each one out as they appear to proceed.
- Next, the model will train. The training progress will be printed to the console.
- After training completes, predictions will be issued on the test data. The test image along with the plotted bounding box prediction will be displayed for the test images. Close them out as they appear to proceed to the next result.
