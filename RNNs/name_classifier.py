import torch 
from torch import nn
import string
import torch.nn.functional as F

allowed_characters = string.ascii_letters + " .,;'" + "_"
n_letters = len(allowed_characters)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RNN(nn.Module):
	def __init__(self, input_size: int, hidden: int, output: int) -> None:
	    super().__init__()
	    self.rnn = nn.RNN(input_size, hidden)
	    self.fc = nn.Linear(hidden, output)

	def forward(self, x):
	    output, hidden = self.rnn(x)
	    x = self.fc(hidden[0])
	    return x

def wordToTensor(word: str) -> torch.Tensor:

    ret = torch.zeros((len(word), 1, n_letters))

    for li, letter in enumerate(word):
        one_hot = torch.zeros(n_letters)
        index = -1
        if letter in allowed_characters:
            index = allowed_characters.find(letter)
        one_hot[index] = 1
        ret[li] = one_hot
    return ret

def outputToLabel(output, labels):
    softmax = F.softmax(output, dim=1)
    return labels[torch.argmax(softmax, dim=1)]

labels = ['Chinese', 'English', 'Scottish', 'Portuguese', 'Greek', 'Russian', 'Vietnamese', 'Polish', 'Dutch', 'French', 'Spanish', 'Irish', 'Italian', 'Arabic', 'Japanese', 'Czech', 'Korean', 'German']


model = RNN(n_letters, 128, 18)
model.load_state_dict(torch.load('nationality_model.pth', map_location=torch.device(device)))
model.eval()

# getting a name for infrence 
name = input("Enter a name: ")
output = model(wordToTensor(name))
prdicete_nationality = outputToLabel(output, labels)
print(f"The predicted nationality is: {prdicete_nationality}")

