# Image Caption Generator

This project implements an image caption generator using a combination of LSTM for text generation and VGG16 for image feature extraction. The model is trained on the Flickr 8k dataset.

## Model Architecture

1. **Image Feature Extraction**: VGG16 pre-trained on ImageNet
2. **Text Generation**: Long Short-Term Memory (LSTM) network

## Dataset

The model is trained on the Flickr 8k dataset, which contains:
- 8,000 images
- 5 captions per image

## Requirements

- Python 3.7+
- TensorFlow 2.0
- Keras
- NumPy
- Matplotlib

## Usage

1. Clone the repository:
`git clone https://github.com/Roronoa-17/Image_Caption_Generator.git`

2. Install the required packages:
`pip install -r requirements.txt`

3. Run the caption generator:
`streamlit run app.py`


## Training

1. Download the Flickr 8k dataset
2. Open the python notebook for further instructions.

