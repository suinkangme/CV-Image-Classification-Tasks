## COMP432 - Group I

<h3>[ Duration ]</h3>
- September 22nd, 2023 to December 5th, 2023 (2023 FALL, Concordia University)


<br>
<br>

<h3>[ Team ]</h3>

| NAME | ID | 
| --- |  --- | 
| Hyun Soo Kim | 40174913 | 
| Matthew Armstrong | 40221458 | 
| Phuong Thao Quach | 26369340 | 
| Suin Kang | 40129337 | 
| Zarren Ali | 40162559 | 

<br>

<h3>[ High level description/presentation of the project ]</h3>

<h4>Convolutional Neural Networks in Image Classification</h4>
<p>
  This project delves into the application of Convolutional Neural Networks (CNNs), a pivotal Machine Learning model in the realm of computer vision, particularly for image classification tasks. CNNs excel in learning and extracting features from images, enabling the accurate classification of new, similar images.
</p>

<h4>Objectives and Tasks</h4>
<p>
  The project is divided into two primary tasks:
</p>
<ol>
  <li>
    <strong>CNN Encoder for Human Tissue Image Classification</strong>: This involves training a CNN encoder on a dataset of human tissue images to classify colon cancer (dataset 1). The outcomes are visualized using t-SNE (t-Distributed Stochastic Neighbor Embedding), a technique for high-dimensional data visualization.
  </li>
  <li>
    <strong>Feature Extraction and Model Evaluation Across Datasets</strong>: Utilizing the trained CNN encoder from the first task and a pre-trained CNN encoder from ImageNet, features are extracted from two additional datasets: a prostate cancer dataset (dataset 2) and an animal faces dataset (dataset 3). The project then focuses on training supervised machine learning models to evaluate and compare the performance of these CNN encoders across diverse datasets.
  </li>
</ol>

<h4>Challenges and Solutions</h4>
<p>
  Training a CNN encoder presents specific challenges, such as the need for large, diverse image datasets and the computational demands of training complex models. To overcome these, strategies like data augmentation and the use of advanced hardware for faster processing are employed.
</p>

<h4>Evaluation Metrics</h4>
<p>
  The performance of the models for both tasks is assessed using key metrics such as precision, recall, f1-score, support, and accuracy.
</p>
<br>

<h3>[ Description on how to obtain the Dataset from an available download link ]</h3>
<p>
  Download links for the datasets required for this assignment are provided below. The first three links lead to the project-required unprocessed data.
  The datasets that follow were generated through feature extraction using both our pretrained ResNet18 model (trained in task 1) and a pretrained Resnet18 model using IMAGENET weights. The final three hyperlinks lead to sampled datasets 1, 2, and 3, each comprising 100 images. The classes are distributed evenly within each class for these sampled datasets (this is an approximation, however, since 100 images must be sampled for three classes, so one class must have one more image).
</p>
<ul>
  <li>
    <a href="https://onedrive.live.com/?authkey=%21ADmb8ZdEzwFMZoo&id=FB338EA7CF297329%21405133&cid=FB338EA7CF297329&parId=root&parQt=sharedby&parCid=UnAuth&o=OneUp">Dataset 1 Original</a>
  </li>
  <li>
    <a href="https://onedrive.live.com/?authkey=%21APy4wecXgMnQ7Kw&id=FB338EA7CF297329%21405132&cid=FB338EA7CF297329&parId=root&parQt=sharedby&parCid=UnAuth&o=OneUp">Dataset 2 Original</a>
  </li>
  <li>
    <a href="https://onedrive.live.com/?authkey=%21AKqEWb1GDjWPbG0&id=FB338EA7CF297329%21405131&cid=FB338EA7CF297329&parId=root&parQt=sharedby&parCid=UnAuth&o=OneUp">Dataset 3 Original</a>
  </li>
  <li>
    <a href="https://drive.google.com/file/d/1bVK8-BYOymcI41wRwsCzx_N65OGCs9vQ/view?usp=sharing">Dataset 2 Extracted by Task 1 Model</a>
  </li>
  <li>
    <a href="https://drive.google.com/file/d/1qxdwp4smOlRc08khxtE6kM0iD_Gfl3AK/view?usp=sharing">Dataset 3 Extracted by Task 1 Model</a>
  </li>
  <li>
    <a href="https://drive.google.com/file/d/1L5mB-u4-rZIt-CRVLEXPr6Li6W0d21ww/view?usp=sharing">Dataset 2 Extracted by ImageNet Model</a>
  </li>
  <li>
    <a href="https://drive.google.com/file/d/1SAkn3JoiVPNxvOQZ97BznVSEcu1fmHDy/view?usp=sharing">Dataset 3 Extracted by ImageNet Model</a>
  </li>
  <li>
    <a href="https://drive.google.com/file/d/1CMR7qOzlY66AFs7xSf4jnpCGqax4CPcA/view?usp=sharing">Sampled Dataset 1</a>
  </li>
  <li>
    <a href="https://drive.google.com/file/d/1UlYZPCoX_rC1lJU8LG3rtYYn-cgzXOTb/view?usp=sharing">Sampled Dataset 2</a>
  </li>
  <li>
    <a href="https://drive.google.com/file/d/1fJljKjN5unR16Qh7HeOsVF3jgYtwIQO-/view?usp=sharing">Sampled Dataset 3</a>
  </li>
</ul>
<br>

<h3>[ Requirements to run your Python code (libraries, etc) ]</h3>
  <p>
  To successfully run the Python code in this repository, several libraries and dependencies need to be installed. The code primarily relies on popular Python libraries such as NumPy, Matplotlib, Pandas, Seaborn, and Scikit-Learn for data manipulation, statistical analysis, and machine learning tasks.
</p>
<p>
  For deep learning models, the code uses PyTorch, along with its submodules such as <code>torchvision</code> and <code>torch.nn</code>. Ensure that you have the latest version of PyTorch installed, which can handle neural networks and various related functionalities.
</p>
<p>
  Additionally, the project uses the <code>Orion</code> library, an asynchronous hyperparameter optimization framework. This can be installed directly from its GitHub repository using the command <code>!pip install git+https://github.com/epistimio/orion.git@develop</code> and its related <code>profet</code> package with <code>!pip install orion[profet]</code>.
</p>
<p>Here is a comprehensive list of all the required libraries:</p>
<ul>
  <li>NumPy</li>
  <li>Pandas</li>
  <li>Matplotlib</li>
  <li>Seaborn</li>
  <li>Scikit-Learn</li>
  <li>PyTorch (along with <code>torch.nn</code>, <code>torch.optim</code>, <code>torch.utils.data</code>, etc.)</li>
  <li>Torchvision (including datasets, models, transforms)</li>
  <li>Orion (including the <code>profet</code> package)</li>
  <li>Argparse (for parsing command-line options)</li>
  <li>TSNE (from Scikit-Learn for dimensionality reduction techniques)</li>
  <li>KNeighborsClassifier, GridSearchCV (from Scikit-Learn for machine learning models)</li>
  <li>RandomForestClassifier (from Scikit-Learn for machine learning models)</li>
  <li>Classification metrics from Scikit-Learn (confusion_matrix, classification_report, etc.)</li>
</ul>
<p>
  For visualization and data analysis, Matplotlib and Seaborn are extensively used. Ensure all these libraries are installed in your environment to avoid any runtime errors.
</p>
<p>
  To install these libraries, you can use pip (Python's package installer). For most libraries, the installation can be as simple as running <code>pip install library-name</code>. For specific versions or sources, refer to the respective library documentation.
</p>

<br>

<h3>[ Instruction on how to train/validate your model ]</h3>
  
 <p>All notebooks were written in Google Colab and are intended for use in Google Colab only. All notebooks are included in our zip submission.</p>
    
  <h4>Task 1: Train the ResNet-18 model from the scratch, Test and Perform t-SNE on Dataset 1</h4>
  <p> Open the notebook - "task1_training_testing.ipynb".</p>
  <ul>
      <li>
          <strong> How To Train? </strong> 
          <ul>
              <li> Run the required libraries</li>
              <li> Run the cell section '1. Data Loading and Preprocessing'</li>
              <li> Run the cell section '2. Training' for training and validation</li>
          </ul>
      </li>
      <br>
      <li>
          <strong> How To Test?</strong>
          <ul>
              <li>No need to upload anything; the test run dataset is available for download via gdown.</li>
              <li>Use the saved model "resnet18_model_98.pth" (which is included in our zip submission).</li>
              <li>Move the pth file to the same directory as the notebook.</li>
              <li>Run the cell section '3. Testing'.</li>
              <li>Run the cell section '4. Feature extracion and t-SNE visualization'.</li>
          </ul>
      </li>
  </ul>
  <br>
  
  <h4>Task 2: Feature Extraction and Classification</h4>
  <ul>
      <li>
          <strong>For Feature Extraction and tSNE:</strong> Run the notebook titled "Task2_Feature_Extraction.ipynb". If you want to save the extracted datasets as csv, run the code under "Save dataset to csv file". If not, leave these code blocks out.
      </li>
      <li>
          <strong>For KNN classification:</strong> Run the notebook titled "Task2_KNN.ipynb".
      </li>
      <li>
          <strong>For RF classification:</strong> Run the notebook titled "Task2_RF.ipynb".
      </li>
  </ul>

<br>

<h3>[ Instructions on how to run the pre-trained model on the provided sample test dataset ]</h3>
  <p> To run the pre-trained models on the provided sample test datasets, follow the instructions below for each notebook: </p>
  <ul>
    <li>
      For task 1, open the notebook titled "task1_training_testing.ipynb", run the code cells one by one following instructions on the below code cells. The instructions in the actual code might differ. If that is the case, follow the instructions in the actual notebook. 
      <br><br>
      <img width="589" alt="image" src="https://github.com/suinkangme/COMP432-GroupI/blob/main/img/sampleset1_1.png">
      <img width="589" alt="image" src="https://github.com/suinkangme/COMP432-GroupI/blob/main/img/sampleset1_2.png">
    </li>
    <li>
      For task 2, open the notebook titled "Task2_Feature_Extraction.ipynb", run the code cells one by one following instructions on the below code cells. The instructions in the actual code might differ. If that is the case, follow the instructions in the actual notebook. 
      <br><br>
      <img width="589" alt="image" src="https://github.com/suinkangme/COMP432-GroupI/blob/main/img/sampleset23_1.png">
      <img width="589" alt="image" src="https://github.com/suinkangme/COMP432-GroupI/blob/main/img/sampleset23_2.png">
      <img width="589" alt="image" src="https://github.com/suinkangme/COMP432-GroupI/blob/main/img/sampleset23_3.png">
    </li>
  </ul>





