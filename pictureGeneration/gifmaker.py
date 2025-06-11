from PIL import Image
import glob
import os

def make_gif(image_folder, output_gif, duration=1000):
    # Get all PNG images in the folder, sorted by name
    images = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    frames = [Image.open(img) for img in images]
    print("Looking for images in:", os.path.abspath(image_folder))
    print("Found files:", images)
    if frames:
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=duration,  # milliseconds per frame (1000ms = 1s)
            loop=1
        )
        print(f"GIF saved to {output_gif}")
    else:
        print("No images found to make GIF.")

# Example usage:
make_gif("pictureGeneration/gifs/epochs/", "pictureGeneration/gifs/generated/animated.gif", duration=1000)