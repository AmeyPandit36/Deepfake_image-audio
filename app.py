import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

# --- 1. MODEL ARCHITECTURES ---

# [AUDIO] GAT Model Architecture
class EfficientGraphAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        return self.norm(x + attn_output)

class SOTA_AudioDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=11, stride=5, padding=5) 
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, stride=3, padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.gat = EfficientGraphAttention(128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.transpose(1, 2)
        x = self.gat(x)
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(2)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# --- 2. CACHED MODEL LOADING (Prevents Crash) ---

@st.cache_resource
def load_audio_model():
    model = SOTA_AudioDetector()
    if os.path.exists("sota_deepfake_detector.pth"):
        model.load_state_dict(torch.load("sota_deepfake_detector.pth", map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_image_model():
    # Assuming VGG16 based on your filename 'deepfake_vgg16_epoch_1.pth'
    model = models.vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, 2) 
    if os.path.exists("deepfake_vgg16_epoch_1.pth"):
        model.load_state_dict(torch.load("deepfake_vgg16_epoch_1.pth", map_location="cpu"))
    model.eval()
    return model

# --- 3. PREPROCESSING ---

def process_audio(file):
    y, sr = librosa.load(file, sr=16000)
    # Remove silence
    y, _ = librosa.effects.trim(y)
    # Standardize to 4 seconds
    y = librosa.util.fix_length(y, size=64000)
    # Peak Norm
    y = y / (np.max(np.abs(y)) + 1e-7)
    return torch.from_numpy(y).unsqueeze(0).unsqueeze(0).float()

def process_image(file):
    img = Image.open(file).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# --- 4. UI ---

st.set_page_config(page_title="Deepfake Shield", layout="wide")
st.sidebar.title("🛡️ Forensic Control")
mode = st.sidebar.radio("Module", ["Audio Lab", "Image Lab"])

if mode == "Audio Lab":
    st.title("🎙️ Audio Forensic Scanner")
    uploaded = st.file_uploader("Upload Audio", type=["wav", "mp3"])
    if uploaded:
        st.audio(uploaded)
        if st.button("DIAGNOSE AUDIO"):
            model = load_audio_model()
            tensor = process_audio(uploaded)
            output = model(tensor)
            prob = torch.softmax(output, dim=1)[0][1].item()
            
            st.subheader("Result Analysis")
            if prob > 0.85:
                st.error(f"### FINAL VERDICT: **FAKE**")
                st.write(f"Confidence: {prob*100:.2f}% (Synthetic signature detected)")
            else:
                st.success(f"### FINAL VERDICT: **REAL**")
                st.write(f"Confidence: {(1-prob)*100:.2f}% (Human speech profile)")

elif mode == "Image Lab":
    st.title("🖼️ Image Forensic Scanner")
    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded:
        st.image(uploaded, width=400)
        if st.button("DIAGNOSE IMAGE"):
            model = load_image_model()
            tensor = process_image(uploaded)
            output = model(tensor)
            prob = torch.softmax(output, dim=1)[0][1].item()
            
            st.subheader("Result Analysis")
            if prob > 0.5:
                st.error(f"### FINAL VERDICT: **FAKE**")
                st.write(f"Confidence: {prob*100:.2f}% (AI-generated artifacts found)")
            else:
                st.success(f"### FINAL VERDICT: **REAL**")
                st.write(f"Confidence: {(1-prob)*100:.2f}% (Organic pixel consistency)")
