import os
import torch
import torch.nn as nn
from nltk.tokenize import sent_tokenize, word_tokenize
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertTokenizerFast, BertModel
from flask import Flask

app = Flask(__name__)


class LSTM(nn.Module):
    """Defines the classifier powering the webapp. Can be swapped for any PyTorch compatible multi-class multi-label
        classification model """

    def __init__(self, dimension=400, text_field=None, num_layers=3, dr=0.201060549754906, bidirectional=True, output_size=434):
        super().__init__()
        self.embedding = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        print("Model Init")
        self.embedding.eval()

        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=768,
                            hidden_size=dimension,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.drop = nn.Dropout(p=dr)

        self.fc = nn.Linear(2 * dimension, output_size)

    def encode(self, token_ids_tensor, attention_mask):
        """Given output from BertTokenizer.convert_tokens_to_ids() as tensor, feed into BERT and then return the sum
        of last four hidden layers as a tensor of token embeddings shape (batch_size, max_seq_length, 768)"""
        with torch.no_grad():
            outputs = self.embedding(token_ids_tensor, attention_mask)

        hidden_states = outputs[2]
        embeddings = torch.stack(hidden_states[-4:]).sum(0)
        return embeddings

    def forward(self, text, text_len, attention_mask):
        # I'd really like to build a more robust pipeline so I can pre-encode everything,
        # but it would take a while so we'll see if on-the-fly encoding doesn't kill performance too much
        text_emb = self.encode(text, attention_mask)
        packed_input = pack_padded_sequence(text_emb, text_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)

        text_fea = self.drop(out_reduced)
        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)

        return text_out


def clean_text(text):
    stripped_tokens = [token for token in word_tokenize(text) if token.isalpha() or token == '>']
    return ' '.join(stripped_tokens)


# TODO handle empty/garbage input
def classify(text):
    """Given Photoshop tutorial text as input, returns a nested dict of label probabilities for each sentence in
    input """
    # TODO Wonder if there's a better way of doing this so the model persists across fucn calls. Will see once I get
    #  Heroku running

    sentences = sent_tokenize(text)
    clean_sentences = list(map(clean_text, sentences))
    word_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # tokenize
    _input = word_tokenizer(clean_sentences, padding=True, return_tensors='pt', return_attention_mask=True, return_length=True)

    # pass tokenized text thru model
    model = LSTM(dr=.3)
    state_dict = torch.load(os.path.join('model', 'model.pt'), map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['model_state_dict'])

    model.eval()
    with torch.no_grad():
        model_output = model(_input['input_ids'], _input['length'], _input['attention_mask'])

    #  We start with: A list of command names, a list of sentences, a matrix with
    #  each row corresponding to a sentence and each column corresponding to a label's probability of being
    #  represented in the sentence The list of command names is parallel to the columns of the matrix

    # We want to end with a nested dict with sentences as keys and dicts of label : probability pairs as values
    labels = model_output.topk(3)

    label_indices = labels[0].tolist()
    probabilities = labels[1].tolist()

    with open(os.path.join('resources', 'label_names.txt')) as f:
        command_names = f.read().splitlines()

    output = dict()
    for i, row in enumerate(probabilities):  # TODO vectorize this if possible
        sent = sentences[i]
        output[sent] = {command_names[idx]: label_indices[i][j] for j, idx in enumerate(row)}

    return output
