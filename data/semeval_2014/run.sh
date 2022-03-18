# download preprocessed data
#git clone https://github.com/songyouwei/ABSA-PyTorch.git
# download the json data from https://github.com/howardhsu/BERT-for-RRC-ABSA and place the json data in json_data
# download the original data from http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools
# the official test xml files do not have labels, we use the json file as labels

# other resources
# https://drive.google.com/file/d/1NGH5bqzEx6aDlYJ7O3hepZF4i_p4iMR8/view

mkdir -p brat/train
mkdir -p brat/test
mkdir -p brat/dev
mkdir -p brat/test_lap
mkdir -p brat/test_rest

#python raw2brat.py --inp ABSA-PyTorch/datasets/semeval14/Laptops_Train.xml.seg:ABSA-PyTorch/datasets/semeval14/Restaurants_Train.xml.seg --out brat/train
#python raw2brat.py --inp ABSA-PyTorch/datasets/semeval14/Laptops_Test_Gold.xml.seg:ABSA-PyTorch/datasets/semeval14/Restaurants_Test_Gold.xml.seg --out brat/test
#python raw2brat.py --inp ABSA-PyTorch/datasets/semeval14/Laptops_Test_Gold.xml.seg --out brat/test_lap
#python raw2brat.py --inp ABSA-PyTorch/datasets/semeval14/Restaurants_Test_Gold.xml.seg --out brat/test_rest

python xml2brat.py --inp raw/Laptop_Train_v2.xml::raw/Restaurants_Train_v2.xml --out brat/train
python xml2brat.py --inp raw/Laptops_Test_Data_phaseB.xml:json_data/asc/laptop/test.json::raw/Restaurants_Test_Data_phaseB.xml:json_data/asc/rest/test.json --out brat/test
python xml2brat.py --inp raw/Laptops_Test_Data_phaseB.xml:json_data/asc/laptop/test.json --out brat/test_lap
python xml2brat.py --inp raw/Restaurants_Test_Data_phaseB.xml:json_data/asc/rest/test.json --out brat/test_rest

python ../split.py --inp brat/train/ --out brat/dev/ --ratio 0.2
