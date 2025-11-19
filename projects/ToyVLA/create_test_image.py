"""
Helper script to create test images for the VLA system.
"""
from PIL import Image, ImageDraw, ImageFont
import os

def create_test_scene(output_path: str, scene_type: str = "cubes"):
    """
    Create a simple test scene image.

    Args:
        output_path: Path to save the image
        scene_type: Type of scene to create
    """
    # Create a 800x600 image with white background
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)

    if scene_type == "cubes":
        # Draw a table (brown rectangle)
        draw.rectangle([50, 400, 750, 550], fill='#8B4513', outline='black', width=2)

        # Draw colored cubes/objects on the table
        # Red cube
        draw.rectangle([100, 320, 180, 400], fill='red', outline='black', width=2)

        # Blue sphere (circle)
        draw.ellipse([250, 320, 350, 420], fill='blue', outline='black', width=2)

        # Green block
        draw.rectangle([450, 340, 530, 400], fill='green', outline='black', width=2)

        # Yellow object
        draw.ellipse([600, 350, 680, 420], fill='yellow', outline='black', width=2)

        # Add labels
        try:
            # Try to use a larger font
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()

        draw.text((120, 410), "Red Cube", fill='black', font=font)
        draw.text((275, 430), "Blue Sphere", fill='black', font=font)
        draw.text((460, 410), "Green Block", fill='black', font=font)
        draw.text((610, 430), "Yellow Ball", fill='black', font=font)

    elif scene_type == "bottles":
        # Draw a table
        draw.rectangle([50, 400, 750, 550], fill='#8B4513', outline='black', width=2)

        # Draw bottles
        # Bottle 1
        draw.rectangle([120, 250, 160, 400], fill='darkgreen', outline='black', width=2)
        draw.ellipse([115, 240, 165, 260], fill='darkgreen', outline='black', width=2)

        # Bottle 2
        draw.rectangle([300, 270, 340, 400], fill='brown', outline='black', width=2)
        draw.ellipse([295, 260, 345, 280], fill='brown', outline='black', width=2)

        # Bottle 3
        draw.rectangle([500, 280, 540, 400], fill='blue', outline='black', width=2)
        draw.ellipse([495, 270, 545, 290], fill='blue', outline='black', width=2)

    # Save the image
    img.save(output_path)
    print(f"Created test image: {output_path}")


if __name__ == "__main__":
    # Create test_images directory if it doesn't exist
    os.makedirs("test_images", exist_ok=True)

    # Create different test scenes
    create_test_scene("test_images/scene_cubes.png", "cubes")
    create_test_scene("test_images/scene_bottles.png", "bottles")

    print("\nTest images created successfully!")
    print("You can now run the VLA system and test with these images.")
