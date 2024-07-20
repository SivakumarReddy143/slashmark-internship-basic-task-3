import tkinter as tk
from tkinter import messagebox
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps, ImageGrab

# Load the trained model
model = load_model('model.h5')

# Function to predict the digit
def predict_digit(img):
    # Resize the image to 28x28 pixels
    img = img.resize((28, 28))
    # Convert the image to grayscale
    img = img.convert("L")
    # Convert the image to a numpy array
    img = np.array(img)
    # Normalize the image
    img = img / 255.0
    # Reshape the image to (1, 28, 28, 1)
    img = img.reshape(1, 28, 28, 1)
    # Predict the digit
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

# Function to clear the canvas
def clear_canvas():
    canvas.delete("all")

# Function to get the drawing from the canvas
def get_canvas_image():
    # Update the canvas to make sure it's current
    canvas.update()
    # Get the canvas coordinates
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    # Capture the canvas content
    img = ImageGrab.grab().crop((x, y, x1, y1))
    return img

# Function to handle the prediction
def predict():
    # Get the canvas image
    img = get_canvas_image()
    # Predict the digit
    digit, acc = predict_digit(img)
    # Show the prediction
    messagebox.showinfo("Prediction", f"Digit: {digit}\nAccuracy: {acc:.2f}")

# Create the main window
root = tk.Tk()
root.title("Digit Recognizer")

# Create the canvas for drawing
canvas = tk.Canvas(root, width=200, height=200, bg="white")
canvas.grid(row=0, column=0, pady=2, sticky=tk.W, columnspan=2)

# Add buttons
btn_predict = tk.Button(root, text="Predict", command=predict)
btn_predict.grid(row=1, column=0, pady=2, padx=2)
btn_clear = tk.Button(root, text="Clear", command=clear_canvas)
btn_clear.grid(row=1, column=1, pady=2, padx=2)

# Function to draw on the canvas
def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)

canvas.bind("<B1-Motion>", paint)

# Run the application
root.mainloop()
