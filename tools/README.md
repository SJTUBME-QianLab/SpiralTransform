This repository holds the codebase, dataset and models for the paper

**Combined Spiral Transformation and Model-driven Multi-modal Deep Learning Scheme for Automatic Prediction of TP53 Mutation in Pancreatic Cancer** 
 Xiahan Chen, Xiaozhu Lin, Xiaohua Qian*

# requied
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
Te detailed parameters can be changed in ```batch_train_mini.py``` 

After training, the weights will be saved in ```./runs``` folder

The evaluation results will be saved in ```classification_LinearRegression_result.txt``` file including acc, recall, precision, f1, sensitivity and specificity

# Citation
Please cite the following paper if you use this repository in your reseach.
```
@inproceedings{
  title     = {Combined Spiral Transformation and Model-driven Multi-modal Deep Learning Scheme for Automatic Prediction of TP53 Mutation in Pancreatic Cancer},
  author    = {Xiahan Chen, Xiaozhu Lin, Xiaohua Qian*},
  booktitle = {},
  year      = {},
}
```

# Contact
For any question, feel free to contact
```
Xiahan Chen : chenxiahan@sjtu.edu.cn
```