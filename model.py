import os
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset
from dp_optimizer import DPAdam
from batch_samplers import get_data_loaders
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, BertTokenizer
import datetime

PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index

LEARNING_RATE_MODEL = 1e-6
WARMUP_STEPS = 0
GRADIENT_ACCUMULATION_STEPS = 1
MAX_GRAD_NORM = 1.0
SEED = 42
NO_CUDA = False


class SentimentBERT:
	model = None
	tokenizer = None

	def __init__(self):
		self.pad_token_label_id = PAD_TOKEN_LABEL_ID
		self.device = torch.device("cuda" if torch.cuda.is_available() and not NO_CUDA else "cpu")

	def predict(self, dataloader):
		if self.model is None or self.tokenizer is None:
			self.load()

		preds = self._predict_tags_batched(dataloader)
		return preds

	def evaluate(self, dataloader, n):
		y_pred = self._predict_tags_batched(dataloader)
		y_true = np.append(np.zeros(n), np.ones(n))

		return classification_report(y_true, y_pred)

	def _predict_tags_batched(self, dataloader):
		preds = []
		self.model.eval()
		for batch in tqdm(dataloader, desc="Computing NER tags"):
			batch = tuple(t.to(self.device) for t in batch)

			with torch.no_grad():
				outputs = self.model(batch[0])
				_, is_neg = torch.max(outputs[0], 1)
				preds.extend(is_neg.cpu().detach().numpy())

		return preds

	def train(self, tokenizer, dataloader, model, epochs, output_dir):
		assert self.model is None  # make sure we are not training after load() command
		model.to(self.device)
		self.model = model
		self.tokenizer = tokenizer

		t_total = len(dataloader) // GRADIENT_ACCUMULATION_STEPS * epochs

		optimizer = DPAdam(
			params=model.parameters(),
			l2_norm_clip=1.0,
			noise_multiplier=0,
			minibatch_size=16,
			microbatch_size=1,
			lr=LEARNING_RATE_MODEL,
		)

		scheduler = get_linear_schedule_with_warmup(
			optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=t_total)

		# Train!
		print("***** Running training *****")
		print(f"Training on {len(dataloader)} examples")
		print(f"Num Epochs = {epochs}")
		print(f"Total optimization steps = {t_total}")
		minibatch_loader, microbatch_loader = get_data_loaders(16, 1, 14000)
		global_step = 0
		tr_loss, logging_loss = 0.0, 0.0
		model.zero_grad()
		train_iterator = trange(epochs, desc="Epoch")
		self._set_seed()
		epoch = 0
		for _ in train_iterator:
			epoch += 1
			epoch_iterator = tqdm(dataloader, desc="Iteration")

			for step, batch in enumerate(epoch_iterator):
				model.train()
				count = 0
				for x_micro, y_micro in microbatch_loader(TensorDataset(batch[0], batch[1])):
					outputs = model(x_micro, labels=y_micro)
					loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

					if GRADIENT_ACCUMULATION_STEPS > 1:
						loss = loss / GRADIENT_ACCUMULATION_STEPS
					optimizer.zero_microbatch_grad()
					loss.backward()

					tr_loss += loss.item()
					count += 1

				if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
					torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
					optimizer.step()
					scheduler.step()  # Update learning rate schedule

					model.zero_grad()
					global_step += 1

			save_directory = output_dir + datetime.datetime.now().strftime("%m-%d-%Y") + "/" + str(epoch) + "/"
			os.makedirs(save_directory, exist_ok=True)
			model.save_pretrained(save_directory)
			tokenizer.save_pretrained(save_directory)
		self.model = model

		return global_step, tr_loss / global_step

	def _set_seed(self):
		torch.manual_seed(SEED)
		if self.device == 'gpu':
			torch.cuda.manual_seed_all(SEED)

	def load(self, model_dir='weights/'):
		if not os.path.exists(model_dir):
			raise FileNotFoundError("folder `{}` does not exist. Please make sure weights are there.".format(model_dir))

		self.tokenizer = BertTokenizer.from_pretrained(model_dir)
		self.model = BertForSequenceClassification.from_pretrained(model_dir)
		self.model.to(self.device)
