import streamlit as st
from streamlit_option_menu import option_menu
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim

from PIL import Image

from torch.utils.data import random_split
from torchvision import transforms

from PIL import Image


label_map={
    0:"Chickenpox",
    1:"Measles",
    2:"Monkeypox",
    3:"Normal"
}
classes = ('Chickenpox', 'Measles', 'Monkeypox', 'Normal')
PATH = r"C:/Users/Lenovo/Desktop/tejas/Pox_Detector/resnet18_net.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                     transforms.Resize((64,64)),
                                     transforms.ToTensor()])
def load_model():
	'''
	load a model 
	by default it is resnet 18 for now
	'''
	model = models.resnet18(pretrained=True)
	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, len(classes))
	model.to(device)

	model.load_state_dict(torch.load(PATH,map_location=device))
	model.eval()
	return model

def predict(model, image):
	'''
	pass the model and image url to the function
	Returns: a list of pox types with decreasing probability
	'''
	picture = Image.open(image)
	# Convert the image to grayscale and other transforms
	image = data_transform(picture)
	# store in a list of images
	image = image.to(device)
	images=image.reshape(1,1,64,64)
 
	model.to(device)

    # Ensure model is in evaluation mode
	model.eval()

	new_images = images.repeat(1, 3, 1, 1)
	new_images = new_images.to(device)

	outputs=model(new_images)
	
	# get prediction
	_, predicted = torch.max(outputs, 1)
	ranked_labels=torch.argsort(outputs,1)[0]
	# get all classes in order of probability
	
	
	
	probabilities = torch.softmax(outputs, dim=1)

	# Get the index of the class with the highest probability
	predicted_index = torch.argmax(probabilities).item()
	predicted_class = classes[predicted_index]

	# Get the confidence score for the predicted class
	confidence_score = probabilities[0][predicted_index].item()

	return predicted_class, confidence_score

#Frontend using streamlit 

st.markdown('''
# **MonkeyPox Detector**
---
''')

with st.sidebar.header('Upload your Image'):
	uploaded_file = st.sidebar.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
	st.write("Uploaded Image:")
	st.image(uploaded_file ,width=400,use_column_width=False , channels="BGR")
	model=load_model()
	predicted_class, confidence_score = predict(model, uploaded_file)
	if predicted_class == "Monkeypox":
		st.warning("### Result: MonkeyPox detected")
		st.warning('Chances of MonkeyPox are : {:.2f}%'.format(confidence_score * 100))
	else:
		st.success("### Result : Monkeypox not detected")
else:
	st.write("### Please Upload an image")

    