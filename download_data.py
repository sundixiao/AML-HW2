import urllib.request

train_target_url = 'https://courses.engr.illinois.edu/cs498aml/sp2019/homeworks/train.txt'
urllib.request.urlretrieve(train_target_url, "train_data.txt")


test_target_url = 'https://courses.engr.illinois.edu/cs498aml/sp2019/homeworks/test.txt'
urllib.request.urlretrieve(test_target_url, "test_data.txt")