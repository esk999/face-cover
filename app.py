import streamlit as st
import dlib
import cv2
import numpy as np
from PIL import Image

# 絵文字の選択肢
emoji_options = {
    '😊': 'data/smile.png',
    '😄': 'data/laugh.png',
    '😯': 'data/surprize.png',
    '😍': 'data/heart_eyes.png'
}


# Dlibの顔検出器をロード
detector = dlib.get_frontal_face_detector()

# Streamlitのタイトル
st.title('顔を絵文字で隠せるアプリ')


# ドロップダウンメニューで絵文字を選択
selected_emoji = st.selectbox("Choose an emoji to place:", list(emoji_options.keys()))


# ファイルアップローダー
uploaded_file = st.file_uploader("画像を選択してください", type=["jpg", "jpeg", "png"])

def place_emoji_on_face(image, emoji_path):
    # 画像をRGBで読み込む
    image = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)

    # 絵文字画像を読み込む
    emoji_img = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
    if emoji_img is None:
        raise ValueError("Failed to load emoji image. Check the file path.")

    # BGRからRGBに変換
    if emoji_img.shape[2] == 4:  # アルファチャンネルが存在する場合
        emoji_img[:, :, :3] = cv2.cvtColor(emoji_img[:, :, :3], cv2.COLOR_BGR2RGB)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        emoji_resized = cv2.resize(emoji_img, (w, h))

        for i in range(h):
            for j in range(w):
                if emoji_resized[i, j][3] != 0:
                    image[y + i, x + j] = emoji_resized[i, j][:3]

    return image

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    try:
        emoji_path = emoji_options[selected_emoji]
        image_with_emoji = place_emoji_on_face(image, emoji_path)
        st.image(image_with_emoji, caption='実行結果', use_column_width=True)
    except Exception as e:
        st.error(f"Error processing image: {e}")