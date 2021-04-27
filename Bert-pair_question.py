import pandas
from transformers import BertTokenizerFast
import torch

from tqdm import tqdm
from torch.utils.data import TensorDataset

# split the data to train and test
from sklearn.model_selection import train_test_split
dataset = pandas.read_csv("train.csv")
max_length =[]


X_train, X_test, y_train, y_test = train_test_split(dataset[["question1", "question2"]], 
                                                    dataset["is_duplicate"], test_size=0.2, random_state=42)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


#
def convert_to_dataset_torch(data: pandas.DataFrame, labels: pandas.Series) -> TensorDataset:
    input_ids = []
    attention_masks = []
    token_type_ids = []
    for _, row in tqdm(data.iterrows(), total=data.shape[0]):
        encoded_dict = tokenizer.encode_plus(row["question1"], row["question2"], max_length=max_length, pad_to_max_length=True, 
                      return_attention_mask=True, return_tensors='pt', truncation=True)  
        input_ids.append(encoded_dict['input_ids'])
        token_type_ids.append(encoded_dict["token_type_ids"])    
        attention_masks.append(encoded_dict['attention_mask'])    
        input_ids = torch.cat(input_ids, dim=0)
        token_type_ids = torch.cat(token_type_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels.values)    
    return TensorDataset(input_ids, attention_masks, token_type_ids, labels)

tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased')
encoded_pair = tokenizer.encode(dataset['question1'][0], dataset['question2'][0])
tokenizer.decode(encoded_pair)
from transformers import BertForSequenceClassification

# Training model
# Load BertForSequenceClassification,
bert_model = BertForSequenceClassification.from_pretrained(
    "bert-large-uncased",num_labels=2, output_attentions=False, output_hidden_states=False)
