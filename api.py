from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import torch
import cv2
import numpy as np
import torchvision.transforms.functional as TF
import io
from model import UNet

app = FastAPI(title="CloudMask AI API")

# Setup Device & Initialize Model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}...")

model = UNet(in_channels=4, out_channels=1).to(device)
checkpoint = torch.load("cloudmask_unet_best.pt", map_location=device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
print(f"✅ Best model loaded! (Epoch: {checkpoint['epoch']}, Loss: {checkpoint['best_loss']:.4f})")

@app.get("/")
def health_check():
    return {"status": "CloudMask AI Backend is running!"}

async def read_image(file: UploadFile):
    """Helper function to read uploaded image into numpy array"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return img

@app.post("/predict")
async def predict(red: UploadFile = File(...), 
                  green: UploadFile = File(...), 
                  blue: UploadFile = File(...), 
                  nir: UploadFile = File(...)):
    try:
        # 1. Read files into memory without saving to disk (Fast execution)
        r = await read_image(red)
        g = await read_image(green)
        b = await read_image(blue)
        n = await read_image(nir)
        
        # Check if any image failed to load
        if any(x is None for x in [r, g, b, n]):
            return JSONResponse(status_code=400, content={"error": "Invalid image file provided."})
            
        # 2. Stack and Convert to Tensor
        img_stacked = np.stack([r, g, b, n], axis=-1)
        image_tensor = TF.to_tensor(img_stacked).unsqueeze(0).to(device) # Shape: (1, 4, H, W)
        
        # 3. Model Inference
        with torch.no_grad():
            output = model(image_tensor)
            # Sigmoid lagakar probabilities nikal rahe hain aur threshold 0.5 set kar rahe hain
            predicted_mask = (torch.sigmoid(output) > 0.5).float()
            
        # 4. Process Output (Calculate stats for Kaushik's frontend)
        mask_np = predicted_mask.squeeze().cpu().numpy()
        total_pixels = mask_np.size
        cloud_pixels = np.count_nonzero(mask_np)
        cloud_percentage = (cloud_pixels / total_pixels) * 100
        
        # Abhi ke liye hum sirf JSON stats bhej rahe hain. 
        # Kaushik ko bolna ki backend usko exactly bata dega kitne % cloud hai.
        return {
            "status": "success",
            "cloud_coverage_percentage": round(cloud_percentage, 2),
            "message": "Mask generated successfully (image return logic can be added later)."
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

    