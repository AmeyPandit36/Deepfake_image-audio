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

# --- 1. AUDIO MODEL (GAT) ---
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

# --- 2. CACHED MODEL LOADERS ---

@st.cache_resource
def load_audio_model():
    model = SOTA_AudioDetector()
    path = "sota_deepfake_detector.pth"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_image_model():
    model = models.vgg16(weights=None)
    # UPDATED: Matching your exact checkpoint shapes (256 hidden units)
    model.classifier[6] = nn.Sequential(
        nn.Linear(4096, 256),  # Changed from 512 to 256
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)     # Changed from 512 to 256
    )
    path = "deepfake_vgg16_epoch_1.pth"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

# --- 3. UI LAYOUT ---

st.set_page_config(page_title="Deepfake Shield Pro", layout="wide")
st.sidebar.title("🛡️ Forensic Control")
mode = st.sidebar.radio("Analysis Type", ["🎙️ Audio Lab", "🖼️ Image Lab"])

if mode == "🎙️ Audio Lab":
    st.title("🎙️ Audio Forensic Scanner")
    uploaded = st.file_uploader("Upload Audio File", type=["wav", "mp3"])
    if uploaded:
        st.audio(uploaded)
        if st.button("RUN AUDIO DIAGNOSTIC"):
            with st.spinner("Analyzing spectral patterns..."):
                model = load_audio_model()
                y, _ = librosa.load(uploaded, sr=16000)
                y = librosa.util.fix_length(y, size=64000)
                tensor = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).float()
                
                output = model(tensor)
                prob = torch.softmax(output, dim=1)[0][1].item()
                
                st.divider()
                if prob > 0.80:
                    st.error("## Verdict: **FAKE / AI-GENERATED**")
                else:
                    st.success("## Verdict: **REAL / HUMAN**")
                
                # Full Parameter Table
                st.subheader("📊 Forensic Parameters")
                st.table({
                    "Parameter": ["Model Architecture", "Confidence Score", "Sample Rate", "Feature Extraction", "Artifact Detection"],
                    "Detail": ["Graph Attention Network", f"{max(prob, 1-prob)*100:.2f}%", "16,000 Hz", "Raw Waveform", "Spectral Inconsistency"]
                })

elif mode == "🖼️ Image Lab":
    st.title("🖼️ Image Forensic Scanner")
    uploaded = st.file_uploader("Upload Image File", type=["jpg", "png", "jpeg"])
    if uploaded:
        st.image(uploaded, width=400)
        if st.button("RUN IMAGE DIAGNOSTIC"):
            with st.spinner("Scanning for pixel artifacts..."):
                model = load_image_model()
                img = Image.open(uploaded).convert('RGB')
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                tensor = transform(img).unsqueeze(0)
                
                output = model(tensor)
                prob = torch.softmax(output, dim=1)[0][1].item()
                
                st.divider()
                if prob > 0.50:
                    st.error("## Verdict: **FAKE / AI-GENERATED**")
                else:
                    st.success("## Verdict: **REAL / ORGANIC**")
                    
                # Full Parameter Table
                st.subheader("📊 Forensic Parameters")
                st.table({
                    "Parameter": ["Architecture", "Hidden Layer Size", "Confidence Score", "Input Resolution", "Detection Method"],
                    "Detail": ["VGG16 (Custom Head)", "256 Neurons", f"{max(prob, 1-prob)*100:.2f}%", "224x224 Pixels", "Spatial Texture Analysis"]
                })
