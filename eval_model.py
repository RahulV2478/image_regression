import torch
import os
import sys
from PIL import ImageDraw, Image
from torchvision import transforms

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from network import load_model, train_network, save_model
from data_loading import RegressionTaskData

class CNN_Eval():
    def __init__(self, model_path:str, n_epochs=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # model_path = os.path.join(script_dir, '../image_regression/model.pth')

        img_size = (3, 1500, 2000)
        if os.path.isfile(model_path): 
            print("Model found")
            self.model = load_model(image_size=img_size, filename=model_path)
        else:
            print("Model not found, training new model")
            self.model = train_network(device=self.device, n_epochs=n_epochs, image_size=img_size)
            save_model(self.model, model_path)

    def predict(self, img_url: str, should_log=False):
        with torch.no_grad():
            # Load the original image
            original_image = Image.open(img_url).convert("RGB")
            original_width, original_height = original_image.size

            # Prepare the image tensor for the model (resize if necessary)
            data = RegressionTaskData(predict_only=True)
            image_tensor = data.prepare_image_from_file(img_url)
            model_input_size = image_tensor.shape[-2:]  # (height, width)
            input_height, input_width = model_input_size

            # Perform prediction
            output = self.model(image_tensor.to(self.device))[0]

            # Rescale the output to the original image size
            scale_x = original_width / input_width
            scale_y = original_height / input_height
            x1 = output[0] * scale_x
            y1 = output[1] * scale_y
            x2 = (output[0] + output[2]) * scale_x  # x + width
            y2 = (output[1] + output[3]) * scale_y  # y + height

            if should_log:
                print(f"Predicted coordinates (scaled): x1={x1}, y1={y1}, x2={x2}, y2={y2}")

            # Draw the rectangle on the original image
            draw = ImageDraw.Draw(original_image)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # Return the modified image and prediction output
        return original_image, output


if __name__ == "__main__":
    eval_model = CNN_Eval(model_path='/Users/ananthkothuri/VSCode/Convergent/HomekyndProject/Homekynd-Convergent/ML/image-regression/model.pth')
    img_url = '/Users/ananthkothuri/VSCode/Convergent/HomekyndProject/Homekynd-Convergent/ML/image-regression/example_dataset/test/images/33.jpg'
    predicted_image, box = eval_model.predict(img_url=img_url, should_log=True)
    predicted_image.show()