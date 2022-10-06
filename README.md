# SD-Script
Small and simple Python script for playing with Stable Diffusion models.

## Requirements
SD-Script has been tested with Python 3.10 and PyTorch 1.12.

## Usage
Grab requirements by running command `pip install -r requirements.txt` or `pip install --user -r requirements.txt`.

Run the script with Python by running command `python sdscript.py` while in same directory (assuming you only have single version of Python installed).

## English commands and default values
* `.read` - Reads a file for inputs, each newline being a separate input. Example: `.read sample.txt	`
    *   Additionally, you can load an image file (.png, .jpg, .jpeg) to use as reference image for stable diffusion.
* `.multiline` - Multiline input, press Ctrl + Z and Enter when done.
* `.stop, .exit, .quit` - Exit from SD-Script.
* `.count` - (1) Amount of images generated for same input. Example: `.count 3`
* `.help, .?` - Prints help text.
* `.inferences` - (50) Amount of inferences per image. Example: `.inferences 100`
* `.width` - (512) Width of the generated image, converted to multiples of 64. Example: `.width 1024`
* `.height` - (512) Height of the generated image, converted to multiples of 64. Example: `.height 1024`
* `.seed` - (-1) Seed for random number generator, for reproducible results. Example: `.seed 1234`
* `.reset` - Clears set reference image.
* `.strength` - (0.75) Strength of the referenced image. Example: `.strength 0.5`
* `.guidance` - (7.5) How much the model should follow the prompt (higher = more). Example: `.guidance 8.5`
* `.magic` - (False) Toggles MagicPrompt model to add detail to your prompts. Example: `.magic`
* `.magiccount` - (1) Amount of MagicPrompt prompts to use. Example: `.magiccount 3`
* `.nsfw` - (False) Toggles NSFW filter. Example: `.nsfw`