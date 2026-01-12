import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms

st.set_page_config(page_title="Knee OA Severity Detector")

st.title("🦴 Knee Osteoarthritis Severity Detection")
st.write("Upload a knee X-ray to predict OA severity (Grade 0–4)")

@st.cache_resource
def load_model():
    model = timm.create_model(
        "swinv2_tiny_window8_256",
        pretrained=False,
        num_classes=5
    )
    model.load_state_dict(torch.load("swinv2_knee_oa.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

uploaded_file = st.file_uploader("Upload Knee X-ray", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(probs).item()

    st.subheader(f"🧠 Predicted OA Grade: {pred}")
    st.write("### Probabilities")
    for i, p in enumerate(probs):
        st.write(f"Grade {i}: {p.item()*100:.2f}%")
