# AGFC_Net

## Dataset
Due to file size limitations, we provide one dataset (DBLP) as an example to test the code. The complete datasets and pre-trained model will be released after the review.

## Environment
+ Python[3.9.12]
+ Pytorch[1.12.0+cu102]
+ GPU[NVIDIA Tesla V100s]

## To run code
+ Step 1: choose the data, i.e., [data_name]=acm/cite/dblp/reut/usps/amap
+ Step 2: python AGFC_Net.py --name [data_name]
* For examle, if you would like to run AGFC_Net on the DBLP dataset, you need to
* run the command "python AGFC_Net.py --name=dblp"
