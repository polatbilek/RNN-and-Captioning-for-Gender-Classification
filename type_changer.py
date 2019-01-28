import os
import imghdr
from PIL import Image
from tqdm import tqdm

path = "/home/cvrg/darg/pan_data/pan18-author-profiling-test-2018-03-20/ar/photo"
total = 0

for user in tqdm(os.listdir(path)):
	photo_file = os.path.join(path,user)
	
	for photo in os.listdir(photo_file):
		image_type = imghdr.what(os.path.join(photo_file, photo))
		im = Image.open(os.path.join(photo_file, photo))

		if image_type != 'jpeg' or im.mode != "RGB":
			print("Found one : " + str(photo))
			
			new_name = str(photo.strip().split(".")[0]) + str(".") + str(photo.strip().split(".")[1]) + str(".jpeg")
			im = im.convert("RGB")

			os.remove(os.path.join(photo_file, photo))
			im.save(os.path.join(photo_file, new_name), "JPEG", quality=100)
			
			total += 1
		
					
print(total)
