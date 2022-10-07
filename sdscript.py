# MIT License
# 
# Copyright (c) 2022 Kristian Ollikainen
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import re
import torch
import random
import pathlib
import argparse
from PIL import Image
from transformers import pipeline, set_seed
from diffusers import DiffusionPipeline, DDIMScheduler, StableDiffusionImg2ImgPipeline

# Version variable
version = "1.0"

# Help string
help_text = """
.read - Reads a file for inputs, each newline being a separate input. Example: .read sample.txt
	  - Additionally, you can load an image file (.png, .jpg, .jpeg) to use as reference image for stable diffusion.
.multiline - Multiline input, press Ctrl + Z and Enter when done.
.stop, .exit, .quit - Exit from SD-Script.
.count - (1) Amount of images generated for same input. Example: .count 3
.help, .? - Prints help text.
.inferences - (50) Amount of inferences per image. Example: .inferences 100
.width - (512) Width of the generated image, converted to multiples of 64. Example: .width 1024
.height - (512) Height of the generated image, converted to multiples of 64. Example: .height 1024
.seed - (-1) Seed for random number generator, for reproducible results. Example: .seed 1234
.reset - Clears set reference image.
.strength - (0.75) Strength of the referenced image. Example: .strength 0.5
.guidance - (7.5) How much the model should follow the prompt (higher = more). Example: .guidance 8.5
.magic - (False) Toggles MagicPrompt model to add detail to your prompts. Example: .magic
.magiccount - (1) Amount of MagicPrompt prompts to use. Example: .magiccount 3
.nsfw - (False) Toggles NSFW filter. Example: .nsfw
"""

# Language strings
languages = \
	{
		"en": {
			"id_language": "en",
			"menu_version": "SD-Script version " + version,
			"menu_info": "Options, '->' means currently selected and '--' means not selected",
			"menu_exit": "Exit",
			"menu_language": "Switch language: English",
			"menu_start": "Start with chosen parameters",
			"menu_usegpu": "Use CUDA (NVIDIA GPU) if available",
			"menu_models": "[Models]",
			"menu_input_option": "Option number: ",
			"error_unknowncommand": "Unknown command",
			"error_unknownfile": "Unknown file",
			"error_requiretoken": "Hugging Face token is required for this model. Please set it with -t or --token.",
			"warn_cudanotavailable": "CUDA device not found, using CPU",
			"info_loading": "Loading..",
			"input_prompt": "Input: ",
			"input_multiline": "v Multiline input, press Ctrl + Z and Enter when done v",
			"info_nsfwtoggle": "NSFW allowed? ",
			"info_magictoggle": "Magic enabled? ",
			"info_inferencesteptoggle": "Inference step images enabled? ",
		},

		"fi": {
			"id_language": "fi",
			"menu_version": "GPT-Script versio " + version,
			"menu_info": "Vaihtoehdot, '->' tarkoittaa tällä hetkellä valittua ja '--' tarkoittaa valitsematonta",
			"menu_exit": "Poistu",
			"menu_language": "Vaihda kieltä: Suomi",
			"menu_start": "Aloita valituilla parametreillä",
			"menu_usegpu": "Käytä CUDA:a (NVIDIA näytönohjain) jos saatavilla",
			"menu_models": "[Mallit]",
			"menu_input_option": "Vaihtoehdon numero: ",
			"error_unknowncommand": "Tuntematon komento",
			"error_unknownfile": "Tuntematon tiedosto",
			"error_requiretoken": "Tämän mallin käyttö vaatii Hugging Face tokenin. Aseta se -t tai --token parametrilla.",
			"warn_cudanotavailable": "CUDA laitetta ei löytynyt, käytetään prosessoria",
			"info_loading": "Ladataan..",
			"input_prompt": "Syöte: ",
			"input_multiline": "v Monirivi syöte, paina Ctrl + Z ja Enter kun valmis v",
			"info_nsfwtoggle": "NSFW sallittu? ",
			"info_magictoggle": "Taikuus käytössä? ",
			"info_inferencesteptoggle": "Inferenssivaiheen kuvat käytössä? ",
		},
	}

# Default language
language = languages["en"]

# Command strings (not restriced by chosen language so they work both ways)
commands = \
	{
		"readfile": (".read", ".lue"),
		"outputcount": (".count", ".määrä"),
		"exitscript": (".stop", ".exit", ".quit", ".poistu", ".sulje", ".lopeta"),
		"multilineinput": (".multiline", ".monirivi"),
		"help": (".help", ".apua", ".?"),
		"inferencecount": (".inferences", ".päättelyt"),
		"width": (".width", ".leveys"),
		"height": (".height", ".korkeus"),
		"guidancescale": (".guidance", ".ohjaus"),
		"rngseed": (".seed", ".siemen"),
		"reset": (".reset", ".nollaa"),
		"imagestrength": (".strength", ".voima"),
		"nsfwtoggle": (".nsfw", ".nsfw"),
		"magictoggle": (".magic", ".taika"),
		"magiccount": (".magiccount", ".taikamäärä"),
		"magicseed": (".magicseed", ".taikasiemen"),
		"inferencesteptoggle": (".inferencesteps", ".päättelyvaiheet"),
	}

# Defaults
useGPUifAvailable = False

# Models (model path/name, if selected, RAM/VRAM usage, requires authentication)
Models = \
	{
		"CompVis/stable-diffusion-v1-4": [True, 8, True],
		"hakurei/waifu-diffusion": [False, 8, False],
	}

# Magic prompt model
MagicPromptModel = "Gustavosta/MagicPrompt-Stable-Diffusion"

# Look for models in current directory subfolders and add them to the list (subholder has model_index.json)
for model in pathlib.Path(".").glob("*"):
	if pathlib.Path(model / "model_index.json").exists():
		Models[str(model)] = [False, 0, False]

# Argument parser
parser = argparse.ArgumentParser(description="SD-Script - A simple script for generating images with Stable-Diffusion")
parser.add_argument("-t", "--token", help="Huggingface token for model", type=str, default=None)

parsed = parser.parse_args()


# Sets specific model as "selected" (True, others False)
def selectModel(modelName):
	global Models
	for k, v in Models.items():
		v[0] = False
	Models[modelName][0] = True


# Gets the first model with True value
def getSelectedModel():
	global Models
	for k, v in Models.items():
		if v[0]:
			return k


while True:
	print("\033[H\033[2J\033[H")
	print(language["menu_version"])
	print(language["menu_info"] + "\n")
	print("[0] " + language["menu_exit"])
	print("[1] " + language["menu_start"])
	print("[2] " + language["menu_language"])
	print(f"[3] {'->' if useGPUifAvailable else '--'} " + language["menu_usegpu"])
	print("\n" + language["menu_models"])

	modelOffset = 4
	idx = modelOffset
	for k, v in Models.items():
		print(
			f"\t[{idx}] {'->' if v[0] else '--'} {k} (~{v[1] if v[1] > 0 else '?'}GB RAM/VRAM) (Requires Hugging Face token: {'✓' if v[2] else '✗'})")
		idx += 1

	try:
		option = int(input("\n" + language["menu_input_option"]))
	except ValueError:
		continue

	if option == 0:
		exit()
	elif option == 1:
		# Check if token is provided if required
		if Models[getSelectedModel()][2] and not parsed.token:
			print(language["error_requiretoken"])
			input()
			continue

		break
	elif option == 2:
		iterator = iter(languages)
		for k in iterator:
			if k == language["id_language"]:
				language = languages[next(iterator, "en")]
	elif option == 3:
		useGPUifAvailable = not useGPUifAvailable
	else:
		for i, k in enumerate(Models):
			if i == option - modelOffset:
				selectModel(k)
				break

# GPU availability check
if useGPUifAvailable and torch.cuda.is_available():
	usingGPU = True
	dataType = torch.float16
	dev = "cuda:0"
else:
	usingGPU = False
	dataType = torch.float32
	dev = "cpu"
	if useGPUifAvailable:
		print(language["warn_cudanotavailable"])

print("\033[H\033[2J\033[H")
print(language["info_loading"])

model = getSelectedModel()


# Function to set DiffusionPipeline as diffuser
def setDiffuser(img2img: bool = False):
	global diffuser
	global model
	global usingGPU
	global dataType
	global dev
	global parsed

	# Clear out CUDA resources before loading new model
	if usingGPU:
		torch.cuda.empty_cache()

	usedDiffuser = DiffusionPipeline if not img2img else StableDiffusionImg2ImgPipeline
	return usedDiffuser.from_pretrained(model,
	                                    revision="fp16" if usingGPU else None,
	                                    torch_dtype=dataType, low_cpu_mem_usage=True,
	                                    use_auth_token=parsed.token if parsed.token else None,
	                                    scheduler=DDIMScheduler(
		                                    beta_start=0.00085,
		                                    beta_end=0.012,
		                                    beta_schedule="scaled_linear",
		                                    clip_sample=False,
		                                    set_alpha_to_one=False)).to(dev)


def createMagicPipe():
	global MagicPromptModel
	global usingGPU
	global dataType
	global dev
	global parsed

	# Clear out CUDA resources before loading new model
	if usingGPU:
		torch.cuda.empty_cache()

	return pipeline("text-generation", model="Gustavosta/MagicPrompt-Stable-Diffusion", tokenizer="gpt2",
	                torch_dtype=dataType, device=dev)


magicPipe = None


def generateMagicPrompt(prompt: str, seed: int = None):
	global magicPipe
	global usingGPU
	global dataType
	global dev

	if not magicPipe:
		return prompt

	if seed is not None:
		set_seed(seed)

	with torch.cuda.amp.autocast(enabled=usingGPU):
		responses = magicPipe(prompt, max_length=(len(prompt) + 100), num_return_sequences=1, pad_token_id=50256)

	result = responses[0]["generated_text"].strip()
	# Clean up prompt
	result = re.sub(r"[^ ]+\.[^ ]+", "", result)
	result = result.replace("<", "").replace(">", "")
	return result


diffuser = setDiffuser(False)
originalSafety = diffuser.safety_checker

print("\033[H\033[2J\033[H")

outCount = 1  # amount of generated images
inferenceCount = 50  # amount of inference steps
guidanceScale = 7.5  # guidance scale
width = 512  # image width
height = 512  # image height
seed = -1  # seed for random number generator
guidanceSourceImage = None  # guidance source image
guidanceImage = None  # guidance image
imageStrength = 0.75  # image strength
nsfwAllowed = False  # allow NSFW images
usingMagicPrompt = False  # use MagicPrompt model to generate stabler images
magicCount = 1  # amount of MagicPrompt variations to generate
magicSeed = -1  # seed for MagicPrompt
usingImg2Img = False  # use image-to-image generation
doingPerInferenceImg = False  # generate image after each inference step


def saveImage(_img, _inferences, _seed):
	saveFile = folder / f"{_seed}_inf{_inferences}.png"

	fileOutCount = 0
	while saveFile.exists():
		saveFile = folder / f"{_seed}_inf{_inferences}_{fileOutCount}.png"
		fileOutCount += 1

	# Save image
	_img.save(saveFile)

while True:
	inText = input(language["input_prompt"])
	if inText.startswith("."):
		if inText.startswith(commands["readfile"]):
			splitted = re.split(r"(\.\w+) (\w+\.\w+)", inText)[2]
			# Use pathlib to check if file exists
			if pathlib.Path(splitted).exists():
				# if text file, read it
				if pathlib.Path(splitted).suffix == ".txt":
					with open(splitted, "r") as f:
						inText = f.read()
				# if image file, use as guidanceImage
				elif pathlib.Path(splitted).suffix in [".png", ".jpg", ".jpeg"]:
					guidanceSourceImage = Image.open(splitted)
					guidanceImage = guidanceSourceImage.convert("RGB").resize((width, height))
					if not usingImg2Img:
						print(language["info_loading"])
						setDiffuser(True)
						usingImg2Img = True
				else:
					print(language["error_unknownfile"])
			else:
				print(language["error_unknownfile"])

			continue
		elif inText.lower().startswith(commands["outputcount"]):
			outCount = int(re.split(r"(\.\w+) (\d+)", inText.lower())[2])
			continue
		elif inText.lower().startswith(commands["reset"]):
			guidanceSourceImage = None
			guidanceImage = None
			if usingImg2Img:
				print(language["info_loading"])
				setDiffuser(False)
				usingImg2Img = False

			continue
		elif inText.lower().startswith(commands["nsfwtoggle"]):
			nsfwAllowed = not nsfwAllowed
			if nsfwAllowed:
				def allowNSFW(images, **kwargs):
					return images, False

				diffuser.safety_checker = allowNSFW
			else:
				diffuser.safety_checker = originalSafety

			print(language["info_nsfwtoggle"] + str(nsfwAllowed))
			continue
		elif inText.lower().startswith(commands["magiccount"]):
			magicCount = int(re.split(r"(\.\w+) (\d+)", inText.lower())[2])
			continue
		elif inText.lower().startswith(commands["magicseed"]):
			magicSeed = int(re.split(r"(\.\w+) (-?\d+)", inText.lower())[2])
			continue
		elif inText.lower().startswith(commands["magictoggle"]):
			usingMagicPrompt = not usingMagicPrompt
			if usingMagicPrompt:
				print(language["info_loading"])
				magicPipe = createMagicPipe()
			else:
				magicPipe = None

			print(language["info_magictoggle"] + str(usingMagicPrompt))
			continue
		elif inText.lower().startswith(commands["inferencesteptoggle"]):
			doingPerInferenceImg = not doingPerInferenceImg
			print(language["info_inferencesteptoggle"] + str(doingPerInferenceImg))
			continue
		elif inText.lower().startswith(commands["inferencecount"]):
			inferenceCount = int(re.split(r"(\.\w+) (\d+)", inText.lower())[2])
			continue
		elif inText.lower().startswith(commands["width"]):
			newWidth = int(re.split(r"(\.\w+) (\d+)", inText.lower())[2])
			# Keep width as multiple of 64
			width = newWidth - (newWidth % 64)
			if usingImg2Img:
				guidanceImage = guidanceSourceImage.convert("RGB").resize((width, height))

			print(width)
			continue
		elif inText.lower().startswith(commands["height"]):
			newHeight = int(re.split(r"(\.\w+) (\d+)", inText.lower())[2])
			# Keep height as multiple of 64
			height = newHeight - (newHeight % 64)
			if usingImg2Img:
				guidanceImage = guidanceSourceImage.convert("RGB").resize((width, height))

			print(height)
			continue
		elif inText.lower().startswith(commands["guidancescale"]):
			guidanceScale = float(re.split(r"(\.\w+) (\d+\.\d+)", inText.lower())[2])
			continue
		elif inText.lower().startswith(commands["imagestrength"]):
			imageStrength = float(re.split(r"(\.\w+) (\d+\.\d+)", inText.lower())[2])
			continue
		elif inText.lower().startswith(commands["exitscript"]):
			break
		elif inText.lower().startswith(commands["multilineinput"]):
			print(language["input_multiline"])
			while True:
				try:
					inText += input() + "\n"
				except EOFError:
					break
		elif inText.lower().startswith(commands["rngseed"]):
			seed = int(re.split(r"(\.\w+) (-?\d+)", inText.lower())[2])
			continue
		elif inText.lower().startswith(commands["help"]):
			print(help_text)
			continue
		else:
			print(language["error_unknowncommand"])
			continue


	for line in inText.splitlines():
		# If using magic prompt, create prompt using line as input
		prompts_seeds = [(line, -1)]
		if usingMagicPrompt:
			# Clear prompts_seeds
			prompts_seeds = []

			# Generate random magic seed if not specified
			curMagicSeed = magicSeed if magicSeed != -1 else random.randint(0, 2 ** 32)

			for i in range(magicCount):
				prompts_seeds.append((generateMagicPrompt(line, curMagicSeed), curMagicSeed))
				# Increment seed
				curMagicSeed += 1

		# Generate image for each prompt
		for prompt, mSeed in prompts_seeds:
			fileOutCount = 0

			# Create folder using prompt with invalid characters removed using regex, limit line name to 50 characters
			folder = pathlib.Path(f"output/{re.sub(r'[^a-zA-Z0-9]+', '', line)[:50]}")
			if usingMagicPrompt:
				folder /= pathlib.Path(f"magic_{mSeed}")
			else:
				folder /= pathlib.Path("nonmagical")

			folder.mkdir(parents=True, exist_ok=True)

			for i in range(outCount):
				# Generate random seed if not specified
				curSeed = seed if seed != -1 else random.randint(0, 2 ** 32)

				# If using static seed, doing multiple outputs and not doing step saving, vary parameters slightly
				gScaleVary = guidanceScale
				iStrengthVary = imageStrength
				if seed != -1 and outCount > 1 and not doingPerInferenceImg:
					gScaleVary += random.uniform(-0.5, 0.5)
					iStrengthVary += random.uniform(-0.25, 0.25)

				# If doing per inference image, generate image for each inference step
				if doingPerInferenceImg:
					for j in range(1, inferenceCount):
						rng = torch.Generator(device=dev).manual_seed(curSeed)

						# Step can't be power of 3, fixed in diffusers v0.4.0 (in dev)
						# TODO: Remove when diffusers v0.4.0 is released
						if j % 3 == 0:
							continue

						with torch.cuda.amp.autocast(enabled=usingGPU):
							saveImage(diffuser(prompt=prompt,
							                   init_image=guidanceImage if usingImg2Img else None,
							                   strength=iStrengthVary if usingImg2Img else None,
							                   guidance_scale=gScaleVary,
							                   width=width, height=height, num_inference_steps=j,
							                   generator=rng).images[0], j, curSeed)
				else:
					rng = torch.Generator(device=dev).manual_seed(curSeed)
					with torch.cuda.amp.autocast(enabled=usingGPU):
						saveImage(diffuser(prompt=prompt, init_image=guidanceImage if usingImg2Img else None,
						                   strength=iStrengthVary if usingImg2Img else None,
						                   guidance_scale=gScaleVary,
						                   width=width, height=height, num_inference_steps=inferenceCount,
						                   generator=rng).images[0], inferenceCount, curSeed)
