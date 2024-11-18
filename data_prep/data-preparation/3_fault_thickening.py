from PIL import Image, ImageFilter
import os


for name in os.listdir('../../data/aug_raw_fault'):
    if name == ".ipynb_checkpoints":
        continue
    if "_" not in name:
        continue
    img = Image.open('../../data/aug_raw_fault/' + name)
    img = img.filter(ImageFilter.GaussianBlur(radius=4))
    img = img.convert("L")
    img = img.point( lambda p: 255 if p > 250 else 0 )
    img = img.convert('1')
    # Display the Box Blurred image
    img.save('../../data/aug_raw_fault_filter/' + name)
    print(f"Finished file: {name}")