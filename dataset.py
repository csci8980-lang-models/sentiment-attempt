import re

import numpy as np
import torch
from torch.utils.data import DataLoader

# The bigger, the better. Depends on your GPU capabilities.
BATCH_SIZE = 16
MAX_SENTENCE_LENGTH = 256


class SentimentDataset:

	def __init__(self, tokenizer):
		self.tokenizer = tokenizer

	@staticmethod
	def _rpad(array, n):
		current_len = len(array)
		if current_len > n:
			return array[:n]
		extra = n - current_len
		return array + ([0] * extra)

	def convert_to_embedding(self, sentence):
		tokens = self.tokenizer.tokenize(sentence)
		tokens = tokens[:MAX_SENTENCE_LENGTH - 2]
		bert_sent = self._rpad(self.tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"]), n=MAX_SENTENCE_LENGTH)

		return bert_sent

	def convert_data_to_embeddings(self, sentences_with_labels):
		for sentence, label in sentences_with_labels:
			bert_sent = self.convert_to_embedding(sentence)
			yield torch.tensor(bert_sent), torch.tensor(label, dtype=torch.int64)

	@staticmethod
	def _parse_imdb_line(line):
		line = line.strip().lower()
		line = line.replace("&nbsp;", " ")
		line = re.sub(r'<br(\s\/)?>', ' ', line)
		line = re.sub(r' +', ' ', line)  # merge multiple spaces into one

		return line

	@staticmethod
	def _read_imdb_data(filename, n):
		print("Getting Data")
		data = []
		positive = 0
		negative = 0
		i = 0
		for line in open(filename, 'r', encoding="utf-8"):
			tmp = SentimentDataset._parse_imdb_line(line)
			if i < n:
				positive += 1
				data.append(tmp)
			if i > 24999 - n:
				negative += 1
				data.append(tmp)
			i += 1
		print("DATA LENGTH", len(data))
		print("Positive:", positive, "Negative:", negative)
		return data

	def prepare_dataloader_from_examples(self, examples, sampler=None):
		dataset = list(self.convert_data_to_embeddings(examples))

		sampler_func = sampler(dataset) if sampler is not None else None
		dataloader = DataLoader(dataset, sampler=sampler_func, batch_size=BATCH_SIZE)

		return dataloader

	def prepare_dataloader(self, filename, n, sampler=None):
		data = self._read_imdb_data(filename, n)
		y = np.append(np.zeros(n), np.ones(n))
		sentences_with_labels = zip(data, y.tolist())

		return self.prepare_dataloader_from_examples(sentences_with_labels, sampler=sampler)
