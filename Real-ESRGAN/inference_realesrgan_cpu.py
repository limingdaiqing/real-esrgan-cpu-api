import argparse
import os
import glob
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
import cv2 # Ensure cv2 is imported globally if used later

def main():
    """
    Real-ESRGAN Inference Script.

    This version is modified to explicitly force the use of CPU for all processing,
    and fixes multiple TypeErrors ('fp32', 'alpha_upsampler', 'denoise_strength') 
    by removing incompatible arguments for older Real-ESRGAN library versions.
    
    It also fixes the 'str' object has no attribute 'shape' error by manually loading 
    the image using cv2.imread before calling upsampler.enhance().
    """
    
    # --- IMPORTANT NOTE FOR CPU USAGE ---
    # When running on CPU, especially for large images, keep an eye on the '--tile' parameter.
    # The default '--tile 0' (no tiling) can cause high RAM usage, potentially crashing 
    # the process if the system runs out of memory (OOM). 
    # On CPU, a larger tile size (e.g., -t 1000 or -t 2000) might actually speed up processing 
    # by reducing the overhead of processing many small tiles.
    # --------------------------------------------------------

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="inputs",
        help="Input image or folder. Default: inputs",
    )
    parser.add_argument(
        "-n",
        "--model_name",
        type=str,
        default="RealESRGAN_x4plus",
        help=("Model name. Default: RealESRGAN_x4plus"),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="results",
        help="Output folder. Default: results",
    )

    # Optional arguments
    # Removed: -dn/--denoise_strength argument definition.
    parser.add_argument(
        "-s",
        "--outscale",
        type=float,
        default=4,
        help="The final upsampling scale of the image. Default: 4.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the pre-trained model file.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="out",
        help="Suffix of the restored image. Default: out.",
    )
    parser.add_argument(
        "-t",
        "--tile",
        type=int,
        default=0,
        help="Tile size. 0 for no tile during testing. Default: 0.",
    )
    parser.add_argument(
        "--tile_pad", type=int, default=10, help="Tile padding. Default: 10."
    )
    parser.add_argument(
        "--pre_pad",
        type=int,
        default=0,
        help="Pre padding size at each border. Default: 0.",
    )
    parser.add_argument(
        "--face_enhance",
        action="store_true",
        help="Use GFPGAN to enhance face. Default: False.",
    )
    # The --fp32 argument is kept for command line compatibility, but it is ignored in RealESRGANer instantiation.
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Adoption of fp32 (single precision). Default: False.",
    )
    # Removed the definition of --alpha_upsampler.
    parser.add_argument(
        "--ext", type=str, default="auto", help="Image extension. Default: auto."
    )

    args = parser.parse_args()

    # --- CORE MODIFICATION: FORCING CPU ---
    device = "cpu"
    print(f"Using device: {device}")

    # -------------------------------------

    # Determine model type and load
    if args.model_name == "RealESRGAN_x4plus":  # x4 RRDBNet
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4
    elif args.model_name == "RealESRGAN_x4plus_anime_6B":  # x4 RRDBNet for anime
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4
        )
        netscale = 4
    elif args.model_name == "realesr-general-x4v3":  # x4 VGG-style model
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        )
        netscale = 4
    elif (
        args.model_name == "realesr-general-wdn-x4v3"
    ):  # x4 VGG-style model (with noise reduction)
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        )
        netscale = 4
    else:
        # Default case or custom model handling
        print(f"Unknown model name: {args.model_name}. Using default RRDBNet x4.")
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4

    # The model path logic remains the same (loading .pth file)
    if args.model_path is not None:
        model_path = args.model_path
    else:
        # Assumes models are in the 'weights' directory
        root_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(root_dir, "weights", args.model_name + ".pth")

    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Exiting.")
        return

    # Create RealESRGANer instance
    # FIX: Removed incompatible arguments
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        # Force device to CPU here
        device=device,
    )

    # If face enhancement is requested, initialize GFPGAN
    if args.face_enhance:
        try:
            from gfpgan import GFPGANer
        except ImportError:
            print(
                "Face enhancement skipped: GFPGAN is not installed. Please install it with: pip install gfpgan"
            )
            args.face_enhance = False

        if args.face_enhance:
            # GFPGAN uses a dedicated model. We keep it on the same device.
            # Assuming GFPGAN weights are also accessible relative to the script
            gfpgan_root_dir = os.path.join(root_dir, 'gfpgan') # Adjust this path if necessary
            gfpgan_model_path = os.path.join(
                gfpgan_root_dir, "weights", "GFPGANv1.4.pth"
            )
            
            if not os.path.exists(gfpgan_model_path):
                 print(f"Warning: GFPGAN model not found at {gfpgan_model_path}. Face enhancement disabled.")
                 args.face_enhance = False
            else:
                face_enhancer = GFPGANer(
                    model_path=gfpgan_model_path,
                    upscale=args.outscale,
                    arch="clean",
                    channel_multiplier=2,
                    bg_upsampler=upsampler,
                )
                print("Face enhancement enabled.")

    # --- Input and Output Handling ---
    if os.path.isdir(args.input):
        # Handle cases where the input is a folder, searching for common image extensions
        #img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp']
        img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp', '*.tiff', '*.jfif', '*.gif']
        paths = []
        for ext in img_extensions:
            paths.extend(glob.glob(os.path.join(args.input, ext)))
        paths = sorted(paths)
    else:
        paths = [args.input]

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    if not paths:
        print(f"No images found at input path: {args.input}. Exiting.")
        return

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print(f"Processing {idx+1}/{len(paths)}: {imgname}{extension} ...")

        try:
            # --- FIX for 'str' object has no attribute 'shape' ---
            # Manually load the image using cv2.imread before passing it to enhance.
            # cv2.IMREAD_UNCHANGED ensures that the alpha channel (if any) is preserved.
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Error: Could not read image file at {path}. Skipping.")
                continue
            # ----------------------------------------------------

            # Read and process image
            if args.face_enhance:
                # GFPGANer enhance method often accepts the path string and handles loading internally.
                _, _, restored_img = face_enhancer.enhance(
                    path, has_aligned=False, only_center_face=False, paste_back=True
                )
            else:
                # RealESRGANer handles the enhancement, passing the loaded image (img) instead of the path
                restored_img, _ = upsampler.enhance(img, outscale=args.outscale)

            # Save image
            if args.ext == "auto":
                extension = extension
            else:
                # Ensure the extension starts with a dot if a custom extension is provided
                extension = "." + args.ext.lstrip('.') 

            save_path = os.path.join(args.output, f"{imgname}_{args.suffix}{extension}")

            # cv2.imwrite is used for image saving
            cv2.imwrite(save_path, restored_img)

            print(f"Saved to {save_path}")

        except RuntimeError as error:
            print(f"Error processing {imgname} (RuntimeError): {error}")
        except Exception as error:
            print(f"Error processing {imgname} (Generic Error): {error}")


if __name__ == "__main__":
    main()