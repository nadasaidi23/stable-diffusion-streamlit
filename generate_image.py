from diffusers import StableDiffusionPipeline
import torch

# Load the model from Hugging Face (CPU mode)
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token="insert_your_token_here",  # paste the token here
    
    torch_dtype=torch.float32
)

pipe = pipe.to("cpu")  # Force CPU mode

prompt = "a cat drinking coffee in a cozy cafe, digital art style"
image = pipe(prompt).images[0]
image.save("generated_image.png")

print("✅ Image saved as 'generated_image.png'")
