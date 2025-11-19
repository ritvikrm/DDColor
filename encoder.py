import os 
import argparse
import subprocess
import cv2
import numpy as np
from PIL import Image
import lzma
import pickle

def convert_to_gray(input_folder, gray_folder):
    for fname in os.listdir(input_folder):
        if fname.lower().endswith(".jpg"):
            inpath = os.path.join(input_folder, fname)
            outpath = os.path.join(gray_folder, fname)
            try:
                img = Image.open(inpath).convert("L")
                img.save(outpath)
                print(f"Converted {fname}")
            except Exception as e:
                print(f"Failed to convert {fname}: {e}")

def run_infer(gray_dir):
    command = [
        "python", "infer.py",
        "--model_path", "./modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt",
        "--input", gray_dir
    ]
    print("About to run infer")
    subprocess.run(command)

def calculate_error(originals_folder, recreation_folder, output_folder):
    for fname in os.listdir(originals_folder):
        original_path = os.path.join(originals_folder, fname)
        recreated_path = os.path.join(recreation_folder, fname)

        try:
            orig = np.array(Image.open(original_path).convert("RGB"), dtype=np.float32)
            rec  = np.array(Image.open(recreated_path).convert("RGB"), dtype=np.float32)

            # Extract RGB channels
            Ro, Go, Bo = orig[...,0], orig[...,1], orig[...,2]
            Rr, Gr, Br = rec[...,0],  rec[...,1],  rec[...,2]

            # Convert to YUV
            Yo = 0.299*Ro + 0.587*Go + 0.114*Bo
            Uo = -0.299*Ro - 0.587*Go + 0.886*Bo
            Vo =  0.701*Ro - 0.587*Go - 0.114*Bo

            Yr = 0.299*Rr + 0.587*Gr + 0.114*Br
            Ur = -0.299*Rr - 0.587*Gr + 0.886*Br
            Vr =  0.701*Rr - 0.587*Gr - 0.114*Br

            # Residuals
            U_resid = Uo - Ur
            V_resid = Vo - Vr

            scale = 4.0
            Uq = np.clip(np.round(U_resid / scale), -128, 127).astype(np.int8)
            Vq = np.clip(np.round(V_resid / scale), -128, 127).astype(np.int8)

            Uq_ds = Uq[::2, ::2]
            Vq_ds = Vq[::2, ::2]

            pack = {
                "Y": Yo,
                "U_resid_q": Uq_ds,
                "V_resid_q": Vq_ds,
                "scale": scale
            }            

            # Save compressed residuals
            out_path = os.path.join(output_folder, fname.replace(".jpg", ".lzma"))

            with lzma.open(out_path, "wb", preset=9 | lzma.PRESET_EXTREME) as f:
                pickle.dump(pack, f, protocol=4)

            print(f"Saved residuals for {fname}")

        except Exception as e:
            print(f"Error with {fname}: {e}")





def main():
    convert_to_gray("Originals", "Grays")
    run_infer("Grays")
    calculate_error("Originals", "results", "Residuals")
    
    
if __name__ == "__main__":
    main()
