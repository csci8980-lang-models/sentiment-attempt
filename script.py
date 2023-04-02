import os
import argparse
import random
from torch.utils.data import RandomSampler
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
import datetime
import torch

from dataset import SentimentDataset
from model import SentimentBERT

BERT_MODEL = 'bert-base-uncased'
NUM_LABELS = 2  # negative and positive reviews
PORTION = .9

parser = argparse.ArgumentParser(prog='script')
parser.add_argument('--train', action="store_true", help="Train new weights")
parser.add_argument('--paramF', action="store_true", help="Freeze subset of layers")
parser.add_argument('--layerF', action="store_true", help="Freeze subset of parameters")
parser.add_argument('-n', type=int, help="Dataset size")
parser.add_argument('--epoch', type=int, help="Num Epochs")
parser.add_argument('--freeze', action="store_true", help="Freeze bert")
parser.add_argument('--evaluate', action="store_true", help="Evaluate existing weights")
parser.add_argument('--predict', default="", type=str, help="Predict sentiment on a given sentence")
parser.add_argument('--path', default='weights/', type=str, help="Weights path")
parser.add_argument('--train-file', default='data/imdb_train.txt',
					type=str, help="IMDB train file. One sentence per line.")
parser.add_argument('--test-file', default='data/imdb_test.txt',
					type=str, help="IMDB train file. One sentence per line.")
args = parser.parse_args()


def train(train_file, epochs, output_dir, n):
	n = int(n / 2)
	print("Torch CUDA is available:", torch.cuda.is_available())
	config = BertConfig.from_pretrained(BERT_MODEL, num_labels=NUM_LABELS)
	tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
	model = BertForSequenceClassification.from_pretrained(BERT_MODEL, config=config)

	if args.freeze:
		output_dir = output_dir + "freeze/"
		for layer in model.bert.encoder.layer:
			for param in layer.parameters():
				param.requires_grad = False

	if args.layerF:
		output_dir = output_dir + "layerF/"
		layers = [model.fc]
		for layer in model.bert.encoder.layer:
			layers.append(layer)

		count = int(len(layers) * PORTION)
		layers = random.sample(layers, count)

		for layer in layers:
			for param in layer.parameters():
				param.requires_grad = False

	if args.paramF:
		output_dir = output_dir + "paramF/"
		parameters = []
		for layer in model.bert.encoder.layer:
			for param in layer.parameters():
				if param.requires_grad:
					parameters.append(param)

		for param in model.parameters():
			if param.requires_grad:
				parameters.append(param)

		count = int(len(parameters) * PORTION)
		subset = random.sample(parameters, count)

		for param in subset:
			param.requires_grad = False

	save_dir = output_dir + datetime.datetime.now().strftime("%m-%d-%Y") + "/final/"
	os.makedirs(save_dir, exist_ok=True)
	with open(save_dir + "time.txt", 'w+') as f:
		f.write(datetime.datetime.now().strftime("Start: %m-%d-%Y %H:%M:%S\n"))
	dt = SentimentDataset(tokenizer)
	dataloader = dt.prepare_dataloader(train_file, n, sampler=RandomSampler)
	predictor = SentimentBERT()
	predictor.train(tokenizer, dataloader, model, epochs, output_dir)
	with open(save_dir + "time.txt", 'a') as f:
		f.write(datetime.datetime.now().strftime("Finish: %m-%d-%Y %H:%M:%S\n"))
	model.save_pretrained(save_dir)
	tokenizer.save_pretrained(save_dir)


def evaluate(test_file, model_dir, n):
	n = int(n / 2)
	predictor = SentimentBERT()
	predictor.load(model_dir=model_dir)

	dt = SentimentDataset(predictor.tokenizer)
	dataloader = dt.prepare_dataloader(test_file, n)
	score = predictor.evaluate(dataloader, n)
	print("SCORE!")
	filename = model_dir + datetime.datetime.now().strftime("%m-%d-%Y %H %M") + ".txt"
	with open(filename, 'w+') as f:
		f.write(score)
	print(score)


def predict(text, model_dir):
	predictor = SentimentBERT()
	predictor.load(model_dir=model_dir)

	dt = SentimentDataset(predictor.tokenizer)
	dataloader = dt.prepare_dataloader_from_examples([(text, -1)], sampler=None)  # text and a dummy label
	result = predictor.predict(dataloader)

	return "Positive" if result[0] == 0 else "Negative"


if __name__ == '__main__':

	epochs = args.epoch or 20
	n = args.n or 25000
	path = args.path or "weights/"
	if args.train:
		os.makedirs(args.path, exist_ok=True)
		train(args.train_file, epochs, path, n)

	if args.evaluate:
		evaluate(args.test_file, path, n)

	if len(args.predict) > 0:
		print(predict(args.predict, args.path))
