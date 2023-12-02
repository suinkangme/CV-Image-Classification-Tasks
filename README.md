## COMP432 - Group I

<h3>[ Duration ]</h3>
- September 22nd, 2023 to December 5th, 2023 (2023 FALL, Concordia University)


<br>
<br>

<h3>[ Team ]</h3>

| NAME | ID | 
| --- |  --- | 
| Hyun Soo Kim |  | 
| Matthew Armstrong |  | 
| Phuong Thao Quach | 26369340 | 
| Suin Kang |  | 
| Zarren Ali |  | 

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
    <strong>Feature Extraction and Model Evaluation Across Datasets</strong>: Utilizing the trained CNN encoder from the first task and a pre-trained CNN encoder from ImageNet, features are extracted from two additional datasets: a prostate cancer dataset (dataset 2) and an animal faces dataset (dataset 3). The project then focuses on training both unsupervised and supervised machine learning models to evaluate and compare the performance of these CNN encoders across diverse datasets.
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
  The datasets that follow were generated through feature extraction using both our pretrained ResNet18 model (trained in task 1) and a pretrained Resnet18 model using IMAGENET weights.
</p>
<ul>
  <li>
    <a href="https://onedrive.live.com/?authkey=%21ADmb8ZdEzwFMZoo&id=FB338EA7CF297329%21405133&cid=FB338EA7CF297329&parId=root&parQt=sharedby&parCid=UnAuth&o=OneUp">Dataset1 Original</a>
  </li>
  <li>
    <a href="https://onedrive.live.com/?authkey=%21APy4wecXgMnQ7Kw&id=FB338EA7CF297329%21405132&cid=FB338EA7CF297329&parId=root&parQt=sharedby&parCid=UnAuth&o=OneUp">Dataset2 Original</a>
  </li>
  <li>
    <a href="https://onedrive.live.com/?authkey=%21AKqEWb1GDjWPbG0&id=FB338EA7CF297329%21405131&cid=FB338EA7CF297329&parId=root&parQt=sharedby&parCid=UnAuth&o=OneUp">Dataset3 Original</a>
  </li>
  <li>
    <a href="https://drive.google.com/file/d/1bVK8-BYOymcI41wRwsCzx_N65OGCs9vQ/view?usp=sharing">Dataset2 Extracted by Task 1 Model</a>
  </li>
  <li>
    <a href="https://drive.google.com/file/d/1qxdwp4smOlRc08khxtE6kM0iD_Gfl3AK/view?usp=sharing">Dataset3 Extracted by Task 1 Model</a>
  </li>
  <li>
    <a href="https://drive.google.com/file/d/1L5mB-u4-rZIt-CRVLEXPr6Li6W0d21ww/view?usp=sharing">Dataset2 Extracted by ImageNet Model</a>
  </li>
  <li>
    <a href="https://drive.google.com/file/d/1SAkn3JoiVPNxvOQZ97BznVSEcu1fmHDy/view?usp=sharing">Dataset3 Extracted by ImageNet Model</a>
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
  <li>Matplotlib</li>
  <li>Pandas</li>
  <li>Seaborn</li>
  <li>Scikit-Learn</li>
  <li>PyTorch (along with <code>torch.nn</code>, <code>torch.optim</code>, <code>torch.utils.data</code>, etc.)</li>
  <li>Torchvision (including datasets, models, transforms)</li>
  <li>Orion (including the <code>profet</code> package)</li>
  <li>Argparse (for parsing command-line options)</li>
  <li>TSNE (from Scikit-Learn for dimensionality reduction techniques)</li>
  <li>KNeighborsClassifier, GridSearchCV (from Scikit-Learn for machine learning models)</li>
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
  
-

<br>

<h3>[ Instructions on how to run the pre-trained model on the provided sample test dataset ]</h3>
  
-





