# JSSI
Joint Shared-and-Specific Information for Deep Multi-View Clustering

## Methodology
The framework of our proposed JSSI, which includes a shared-and-specific feature extraction network, and a feature representation and clustering network. The shared-and-specific feature extraction network (yellow dashed box), as the name suggests, separates the shared and specific information using an adversarial similarity constraint, a shared-and-specific space difference constraint and a reconstruction constraint. The feature representation and clustering network (green dashed box) concatenates the specific features and aligns the shared feature to generate a new representation for K-Means clustering.
![image](https://github.com/Chenjin8249/JSSI/assets/77652938/4ad1ebec-222d-4774-94da-5b24a642db90)

## Datasets
The multi-view dataset for testing is located in the "data" folder, and the dataset for training needs to be extracted from the COCO dataset using the tools in the "data_tool" folder.

## Requirements
numpy==1.23.5
torchvision==0.15.2
torch>=2.0.1

## Usage
The network structure and training/testing pipeline are in ACmodel.py, ACtrain.py and ACtest.py.  
Train the model on the multi-view data from the COCO dataset and test the model on the dataset in the "data" folder.

## Citation
>  @ARTICLE{10130402,  
>    author={Chen, Jin and Huang, Aiping and Gao, Wei and Niu, Yuzhen and Zhao, Tiesong},  
>    journal={IEEE Transactions on Circuits and Systems for Video Technology},   
>    title={Joint Shared-and-Specific Information for Deep Multi-View Clustering},   
>    year={2023},  
>    volume={33},  
>    number={12},  
>    pages={7224-7235},  
>    doi={10.1109/TCSVT.2023.3278285}
> }  
