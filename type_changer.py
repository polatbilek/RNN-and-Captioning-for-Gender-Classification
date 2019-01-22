import os
import imghdr
from PIL import Image
from tqdm import tqdm

path = "/mnt/671728fd-b9e2-46ed-b18b-9f45f387f63e/turkish_tweets_dataset/turkish_tweets/photo"
total = 0

for user in tqdm(os.listdir(path)):
	photo_file = os.path.join(path,user)
	
	for photo in os.listdir(photo_file):
		image_type = imghdr.what(os.path.join(photo_file, photo))
		
		if image_type != 'jpeg':
			print("Found one : " + str(photo))
			im = Image.open(os.path.join(photo_file, photo))
			new_name = str(photo.strip().split(".")[0]) + str(".jpeg")
			im = im.convert("RGB")
			im.save(os.path.join(photo_file, new_name), "JPEG", quality=100)
			
			os.remove(os.path.join(photo_file, photo))
			total += 1
			
		else:
			im = Image.open(os.path.join(photo_file, photo))
			new_name = str(photo.strip().split(".")[0]) + str(".jpeg")
			im.save(os.path.join(photo_file, new_name), "JPEG", quality=100)
			os.remove(os.path.join(photo_file, photo))
		
print(total)