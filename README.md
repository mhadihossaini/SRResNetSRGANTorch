## SRResNetSRGANTorch

#### Overview:

The project introduces a deep learning model for photo-realistic single image super-resolution based on  [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) research paper. It utilizes Residual Blocks with Instance Normalization and PReLU activation, along with upsampling layers through PixelShuffle. A Discriminator network critiques image quality. The model minimizes content loss for fidelity and aims for indistinguishable-from-real photographs, advancing super-resolution towards photo-realism.

#### Installation:
Clone the repository:

git clone https://github.com/mhadihossain/SRResNetSRGANTorch.git

Navigate to the project directory:

cd SRResNetSRGANTorch

Install the required dependencies:

pip install -r requirements.txt

#### Usage:
1. Downloading the Dataset:

The download_dataset is responsible for fetching the dataset (tensorflow dataset) and saving it into separate directories for training and testing. This step ensures that the dataset is readily available for further processing without the need for manual intervention.

2. Loading the Dataset:

The load_dataset is designed to load the dataset from the saved directories (train_dataset and test_dataset) into memory for training the model. In Jupyter Notebook (SRResNetSRGANTorch.ipynb) the dataset is just downloaded but not saved in train_dataset and test_dataset. 

3. Model Definition:

The model_definition module contains the architecture of the deep learning model. This includes the layers, activation functions, and any other components necessary to build the model. The structure of the model is crucial for capturing the underlying patterns in the dataset and making accurate predictions.

4. Traning Model
   
The train_model trains the model(GAN) for single-image super-resolution using a custom SRResNet generator and Discriminator. It optimizes images through adversarial and content loss functions, enhancing low-resolution inputs to produce high-fidelity outputs. Trained models are saved for future use.

5. Testing the Model:

The test_model function evaluates the performance of the trained deep learning model using the test dataset. This step involves feeding image into the model and comparing the predicted outputs with the actual. 
