import os
import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image

# Lấy đường dẫn của script hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))

# Tạo đường dẫn đến file model
model_path = os.path.join(current_dir, "MNIST_model.h5")

# Load model đã được huấn luyện
model = load_model(model_path)

def preprocess_image(image):
    # Tiền xử lý ảnh đầu vào cho mô hình
    img = image.resize((28, 28))  # Resize ảnh về kích thước mà mô hình mong đợi
    img = np.array(img)  # Chuyển đổi ảnh thành mảng numpy
    img = img / 255.0  # Chuẩn hóa giá trị pixel về khoảng từ 0 đến 1
    img = np.expand_dims(img, axis=0)  # Thêm chiều batch
    return img

def predict_digit(image):
    # Dự đoán số trên ảnh đầu vào
    preprocessed_img = preprocess_image(image)
    prediction = model.predict(preprocessed_img)
    predicted_digit = np.argmax(prediction)
    return predicted_digit

def main():
    st.title("Ứng dụng Nhận Dạng Chữ Số Viết Tay")

    # Display an image
    image_path = os.path.join(current_dir, "mnist.png")
    st.image(image_path, use_column_width=True)

    uploaded_file = st.file_uploader("Tải Ảnh Lên", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Hiển thị ảnh đã tải lên
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh Đã Tải Lên", use_column_width=True)

        # Dự đoán chữ số
        predicted_digit = predict_digit(image)
        st.write("Dự đoán:", predicted_digit)

if __name__ == "__main__":
    main()
