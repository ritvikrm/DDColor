import numpy as np
from PIL import Image
import os
import subprocess
import shutil
import cv2
import lzma
import pickle


def recreate_images(residual_folder, gray_folder, results_folder):
    os.makedirs(gray_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    # --- 1. Restore grayscale Y images for model inference ---
    for fname in os.listdir(residual_folder):
        if not fname.endswith(".lzma"):
            continue

        residual_path = os.path.join(residual_folder, fname)
        gray_path = os.path.join(gray_folder, fname.replace(".lzma", ".jpg"))

        # Load compressed file
        with lzma.open(residual_path, "rb") as f:
            data = pickle.load(f)

        Y = data["Y"].astype(np.uint8)

        # Save grayscale for neural net input
        Image.fromarray(Y).save(gray_path)

    # --- 2. Run neural network inference ---
    command = [
        "python", "infer.py",
        "--model_path", "./modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt",
        "--input", gray_folder
    ]
    subprocess.run(command)

    # --- 3. Reconstruct final color images with residuals ---
    for fname in os.listdir(residual_folder):
        if not fname.endswith(".lzma"):
            continue

        base = fname.replace(".lzma", "")

        residual_path = os.path.join(residual_folder, fname)

        # Load compressed residuals
        with lzma.open(residual_path, "rb") as f:
            data = pickle.load(f)

        Y = data["Y"].astype(np.float32)
        Uq_ds = data["U_resid_q"]   # downsampled int8
        Vq_ds = data["V_resid_q"]
        scale = data["scale"]

        H, W = Y.shape

        # --- Upsample chroma residuals to full resolution ---
        Uq_ds = Uq_ds.astype(np.float32)
        Vq_ds = Vq_ds.astype(np.float32)

        Uq = cv2.resize(Uq_ds, (W, H), interpolation=cv2.INTER_LINEAR)
        Vq = cv2.resize(Vq_ds, (W, H), interpolation=cv2.INTER_LINEAR)

        # --- Dequantize residuals ---
        U_resid = Uq * scale
        V_resid = Vq * scale

        # --- Load neural-net-generated RGB output ---
        rec_path = os.path.join(results_folder, base + ".jpg")
        rec = np.array(Image.open(rec_path).convert("RGB"), dtype=np.uint8)
        Rr, Gr, Br = rec[..., 0], rec[..., 1], rec[..., 2]

        # Convert NN output to YUV (same transform as in encoding)
        Ur = -0.169 * Rr - 0.331 * Gr + 0.500 * Br
        Vr = 0.500 * Rr - 0.419 * Gr - 0.081 * Br

        # Add residuals
        U_final = Ur + U_resid
        V_final = Vr + V_resid

        # Convert back to RGB
        R_final = Y + 1.13983 * V_final
        G_final = Y - 0.39465 * U_final - 0.58060 * V_final
        B_final = Y + 2.03211 * U_final

        R8 = np.clip(R_final, 0, 255).astype(np.uint8)
        G8 = np.clip(G_final, 0, 255).astype(np.uint8)
        B8 = np.clip(B_final, 0, 255).astype(np.uint8)

        rgb_final = np.stack([R8, G8, B8], axis=-1)
        final_path = os.path.join(results_folder, base + "_final.jpg")
        Image.fromarray(rgb_final).save(final_path)

        print(f"Saved final colorized image with residuals: {final_path}")

def calculate_mse(img1, img2):
    """Compute Mean Squared Error between two images."""
    err = np.mean((img1.astype(np.uint8) - img2.astype(np.uint8)) ** 2)
    return err

def calculate_psnr(img1, img2):
    """Compute PSNR between two images."""
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')  # Images are identical
    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return mse, psnr

def refresh_results():
    results_folder = "results"
    if os.path.exists(results_folder):
        shutil.rmtree(results_folder)
    os.makedirs(results_folder, exist_ok=True)

    results_folder = "Recreated Grays"
    if os.path.exists(results_folder):
        shutil.rmtree(results_folder)
    os.makedirs(results_folder, exist_ok=True)

def main():
    #refresh_results()
    #recreate_images("Residuals", "Recreated Grays", "results")

    for fname in os.listdir("Originals"):
        base = fname.replace(".jpg", "")
        original = os.path.join("Originals", fname)
        recreated = os.path.join("results", base+"_final.jpg")

        img1 = cv2.imread(original)
        img2 = cv2.imread(recreated)
        mse, psnr = calculate_psnr(img1, img2)
        print(f"FILE: {base}")
        print(f"MSE: {mse}, PSNR: {psnr}")



if __name__ == "__main__":
    main()





