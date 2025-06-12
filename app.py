import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# --- CONFIG ---
st.set_page_config(page_title="🎨 AI Image Generator", layout="wide")

# --- DARK MODE STYLING ---
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: #E0E0E0;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stTextInput>div>div>input {
            background-color: #1E1E1E;
            color: #E0E0E0;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 10px;
        }
        .stButton>button {
            background-color: #673AB7;
            color: white;
            font-size: 16px;
            padding: 12px 24px;
            border-radius: 8px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #7E57C2;
        }
        .title {
            font-size: 42px;
            color: #BB86FC;
            font-weight: bold;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            font-size: 18px;
            color: #B0B0B0;
            text-align: center;
            margin-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        token="insert_your_huggingface_token_here",  # replace with your Hugging Face token
        torch_dtype=torch.float32
    ).to("cpu")

pipe = load_model()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 📝 Prompt Settings")
    prompt = st.text_area("Enter your prompt:", "A futuristic city with neon lights at night", height=120)
    if st.button("🎨 Generate Image"):
        st.session_state["run_generation"] = True

    st.markdown("---")
    st.markdown("**💡 Prompt ideas:**")
    st.markdown("- A panda reading a book in a forest\n- An astronaut on a surfboard\n- A fantasy castle in the clouds")

# --- MAIN AREA ---
st.markdown('<div class="title">Stable Diffusion CPU Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Turn your text into stunning artwork – powered by AI</div>', unsafe_allow_html=True)

if "run_generation" in st.session_state and st.session_state["run_generation"]:
    with st.spinner("🧠 Generating your image... please wait..."):
        try:
            image = pipe(prompt).images[0]
            st.image(image, caption="🖼️ Generated Image", width=512)
            image.save("generated_image.png")
            st.success("✅ Image saved as `generated_image.png`")
            st.session_state["run_generation"] = False
        except Exception as e:
            st.error(f"❌ Error generating image: {e}")
