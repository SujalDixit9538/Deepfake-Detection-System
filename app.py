import gradio as gr
import cv2
import numpy as np
import tensorflow as tf

# Load the model (It looks for this exact name)
model = tf.keras.models.load_model('final_deepfake_model.keras')

def scan_image(image):
    if image is None: return "Please upload an image."
    try:
        img_resized = cv2.resize(image, (224, 224))
        img_batch = np.expand_dims(img_resized, axis=0)
        prediction = model.predict(img_batch)[0][0]
        label = "FAKE" if prediction > 0.5 else "REAL"
        conf = prediction if prediction > 0.5 else 1 - prediction
        return f"🛑 {label} IMAGE detected.\nConfidence: {conf * 100:.2f}%"
    except Exception as e: return f"Error: {e}"

def scan_video(video_file):
    if video_file is None: return "Please upload a video file."
    try:
        cap = cv2.VideoCapture(video_file)
        frames = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_count % 30 == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
            frame_count += 1
        cap.release()
        
        if not frames: return "Error: Could not read video."
        
        preds = model.predict(np.array(frames))
        avg = np.mean(preds)
        label = "FAKE" if avg > 0.5 else "REAL"
        conf = avg if avg > 0.5 else 1 - avg
        return f"🛑 {label} VIDEO detected!\nConfidence: {conf * 100:.2f}%"
    except Exception as e: return f"Error: {e}"

# Interface
with gr.Blocks(title="Deepfake Detector") as demo:
    gr.Markdown("# 🕵️‍♂️ Deepfake Detection System")
    with gr.Tab("🖼️ Image Scanner"):
        btn_img = gr.Button("Scan Image")
        out_img = gr.Textbox(label="Result")
        btn_img.click(scan_image, inputs=gr.Image(), outputs=out_img)
    with gr.Tab("🎥 Video Scanner"):
        btn_vid = gr.Button("Scan Video")
        out_vid = gr.Textbox(label="Result")
        btn_vid.click(scan_video, inputs=gr.Video(), outputs=out_vid)

demo.launch()