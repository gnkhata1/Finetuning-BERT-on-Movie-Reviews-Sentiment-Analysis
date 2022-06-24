
#downloading SST dataset
dataset = load_dataset("sst", "default") 
def load_data(dataset): #changing the dataset to SST-2 version 
    train_d = dataset["train"]
    test_d = dataset["test"]
    val_d = dataset["validation"]

    sentence_tr, label_tr = train_d["sentence"], train_d["label"]
    sentence_ts, label_ts = test_d["sentence"], test_d["label"]
    sentence_vl, label_vl = val_d["sentence"], val_d["label"]

    sentence_tr, label_tr = np.asarray(sentence_tr), np.asarray(label_tr)
    sentence_ts, label_ts = np.asarray(sentence_ts), np.asarray(label_ts)
    sentence_vl, label_vl = np.asarray(sentence_vl), np.asarray(label_vl)
    
    #changing labels to SST-2 version
    label_tr, label_ts, label_vl = np.asarray(np.round(label_tr), int), np.asarray(np.round(label_ts), int), np.asarray(np.round(label_vl), int)

    sentence_tr, label_tr = sentence_tr.reshape(-1, 1), label_tr.reshape(-1, 1)
    sentence_ts, label_ts = sentence_ts.reshape(-1, 1), label_ts.reshape(-1, 1)
    sentence_vl, label_vl = sentence_vl.reshape(-1, 1), label_vl.reshape(-1, 1)


    train = np.concatenate((sentence_tr, label_tr), axis=1)
    test = np.concatenate((sentence_ts, label_ts), axis=1)
    val = np.concatenate((sentence_vl, label_vl), axis=1)
    
    np.random.shuffle(train)
    np.random.shuffle(test)
    np.random.shuffle(val)
    
    whole_data = np.concatenate((train, test, val))

    return whole_data

data = load_data(dataset)

#downloading and preprocessong  amazon reviews dataset for movies
dataset = load_dataset("amazon_us_reviews", "Video_v1_00")

def load_data(dataset):
    data = dataset["train"]
    
    label = data["star_rating"]
    train = data["review_body"]
    
    whole_data = list(zip(train, label))

    return whole_data

def create_bi_amazon(whole_data): #changing the dataset to amazon-2
    whole_data = list(whole_data)
    reviews = []
    labels = []
    rows = len(whole_data)
    
    for i in range(rows):
        if whole_data[i][1] <= 2:
            reviews.append(whole_data[i][0])
            labels.append(0)
            
        elif whole_data[i][1] >= 4:
            reviews.append(whole_data[i][0])
            labels.append(1)
    return list(zip(reviews, labels))     