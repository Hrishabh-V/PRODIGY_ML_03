import customtkinter as ctk
from tkinter import filedialog
import joblib
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
# Load the pre-trained model
svm_model = joblib.load('Task3/svm_model.pkl')  # Ensure your model is in the same directory
le = LabelEncoder()
le.classes_ = np.array(['cat', 'dog'])  

# Function to predict the label of an uploaded image
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128)).reshape(1, -1) / 255.0 
    prediction = svm_model.predict(img)
    return le.inverse_transform(prediction)[0]

# Define the app class
class ImageClassifierApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Cat vs Dog Classifier")
        self.geometry("400x300")
        
        # Create UI elements
        self.label = ctk.CTkLabel(self, text="Upload an Image of a Cat or Dog", font=("Arial", 16))
        self.label.pack(pady=20)

        self.upload_button = ctk.CTkButton(self, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.result_label = ctk.CTkLabel(self, text="", font=("Arial", 14))
        self.result_label.pack(pady=20)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            prediction = predict_image(file_path)
            self.result_label.configure(text=f'The predicted label is: {prediction}')  

# Run the app
if __name__ == "__main__":
    app = ImageClassifierApp()
    app.mainloop()
