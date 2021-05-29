This repository holds the PyTorch code for the paper

**Combined Spiral Transformation and Model-driven Multi-modal Deep Learning Scheme for Automatic Prediction of TP53 Mutation in Pancreatic Cancer** 
 

All the materials released in this library can ONLY be used for RESEARCH purposes and not for commercial use.

The authors' institution (Biomedical Image and Health Informatics Lab, School of Biomedical Engineering, Shanghai Jiao Tong University) preserve the copyright and all legal rights of these codes.

# Author List
Xiahan Chen, Xiaozhu Lin, Xiaohua Qian*
# Abstract
Pancreatic cancer is a malignant form of cancer with one of the worst prognoses. The poor prognosis and resistance to therapeutic modalities have been linked to TP53 mutation. Pathological examinations, such as biopsies, cannot be frequently performed in clinical practice; therefore, noninvasive and reproducible methods are desired. However, automatic prediction methods based on imaging have drawbacks such as poor 3D information utilization, small sample size, and ineffectiveness multimodal fusion. In this study, we proposed a model-driven multi-modal deep learning scheme to overcome these challenges. A spiral transformation algorithm was developed to obtain 2D images from 3D data, with the transformed image inheriting and retaining the spatial correlation of the original texture and edge information. The spiral transformation could be used to effectively apply the 3D information with less computational resources and conveniently augment the data size with high quality. Moreover, model-driven items were designed to introduce prior knowledge in the deep learning framework for multi-modal fusion. The model-driven strategy and spiral transformation-based data augmentation can improve the performance of the small sample size. A bilinear pooling module was introduced to improve the performance of fine-grained prediction. The experimental results show that the proposed model gives the desired performance in predicting TP53 mutation in pancreatic cancer, providing a new approach for noninvasive gene prediction. The proposed methodologies of spiral transformation and model-driven deep learning can also be used for the artificial intelligence community dealing with oncological applications. Our source codes with a demon will be released at https://github.com/SJTUBMEQianLab/SpiralTransform.

# required
Our code is based on **Python3.6** There are a few dependencies to run the code. The major libraries we depend are
- PyTorch1.1.0 (http://pytorch.org/)
- tensorboardX 1.8
- numpy 
- tqdm 

# set up
```
pip install -r requirements.txt
```
Attention:
Please run this project on linux.
In different pytorch environment, the model may obtain different results. 

# quickly train and test
Run the ```batch_train_mini.py``` by this command:
```
python batch_train_mini.py
```
The detailed parameters can be changed in ```batch_train_mini.py``` 

After training, the weights will be saved in ```./result``` folder

The evaluation results will be saved in ```classification_LinearRegression_result.txt``` file including acc, recall, precision, f1, sensitivity and specificity

# Citation
Please cite the following paper if you use this repository in your research.
```
@inproceedings{
  title     = {Combined Spiral Transformation and Model-driven Multi-modal Deep Learning Scheme for Automatic Prediction of TP53 Mutation in Pancreatic Cancer},
  author    = {Xiahan Chen, Xiaozhu Lin, Xiaohua Qian*},
  journal   = {IEEE Transactions on Medical Imaging},
  month     = {February}，
  year      = {2021},
}
```

# Contact
For any question, feel free to contact
```
Xiahan Chen : chenxiahan@sjtu.edu.cn
```
