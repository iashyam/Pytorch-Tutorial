import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import requests
import zipfile
import string
import unicodedata

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using {device} device')

def get_data():
	
	link = "https://download.pytorch.org/tutorial/data.zip"
	if not os.path.exists('data/names'):
		print('Downloading the data')
		with open('download_data', 'wb') as f:
			f.write(requests.get(link).content)

		## unzip the file for 
		with zipfile.ZipFile('download_data', 'r')  as f:
			f.extractall('.')

allowed_characters = string.ascii_letters + " .,;'" + "_"
n_letters = len(allowed_characters)

def to_ascii(word: str) ->str:
	'''convert a non ascii words (words with accent) to ascii'''
	return ''.join(
    	c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
		and c in allowed_characters
		)

#one hot encoded tensor for any word
def encodeWord(word: str) -> torch.Tensor:
	ret = torch.zeros((len(word),1, n_letters), device=device)
	for index, letter in enumerate(word):
		letter_tensor = torch.zeros(n_letters)
		if letter in allowed_characters:
			letter_tensor[allowed_characters.index(letter)] = 1
		else:
			letter_tensor[-1] = 1
		ret[index][0] = letter_tensor

	return ret

class customDataset(Dataset):
	
	def __init__(self, base_dir: str):
		pass		


if __name__=="__main__":
	get_data()
	print(encodeWord('Ahn'))
