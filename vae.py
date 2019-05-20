from __future__ import division
import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
from Loggers import Logger
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import subprocess
import make_zinc_dataset_grammar
import zinc_grammar


def get_gpu_memory_map():
	"""Get the current gpu usage.

	Returns
	-------
	usage: dict
		Keys are device ids as integers.
		Values are memory usage as integers in MB.
	"""
	result = subprocess.check_output(
		[
			'nvidia-smi', '--query-gpu=memory.used,memory.free,memory.total',
			'--format=csv'
		])
	# Convert lines into a dictionary
	print(result.decode("utf-8"))

if torch.cuda.is_available():
	print("Using GPU")
	Tensor = torch.cuda.FloatTensor
else:
	print("Using CPU")
	Tensor = torch.FloatTensor

latent_rep_size = 56

class Zinc_dataset(Dataset):
	def __init__(self, isTrain):
		if isTrain:
			one_hot = np.load("one_hot.npy")[5000:]
		else:
			one_hot = np.load("one_hot.npy")[:5000]
		self.one_hot = one_hot.astype(np.float32)
		
	def __getitem__(self, index):
		return self.one_hot[index]
		
	def __len__(self):
		return len(self.one_hot)


class Encoder(torch.nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.conv1 = torch.nn.Conv1d(in_channels=76, out_channels=9, kernel_size=9)
		self.conv2 = torch.nn.Conv1d(in_channels=9, out_channels=9, kernel_size=9)
		self.conv3 = torch.nn.Conv1d(in_channels=9, out_channels=10, kernel_size=11)
		self.fc = torch.nn.Linear(2510, 435)
		self.fcMean = torch.nn.Linear(435, latent_rep_size)
		self.fcLogVar = torch.nn.Linear(435, latent_rep_size)
		if torch.cuda.is_available():
			self.cuda()

	def forward(self, x):
		x = torch.relu(self.conv1(x))
		x = torch.relu(self.conv2(x))
		x = torch.relu(self.conv3(x))
		x = torch.flatten(x, start_dim=1)
		x = torch.relu(self.fc(x))
		mean = self.fcMean(x)
		log_var = self.fcLogVar(x)
		return mean, log_var


class Decoder(torch.nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		self.fc1 = torch.nn.Linear(latent_rep_size, latent_rep_size)
		self.gru = torch.nn.GRU(input_size=latent_rep_size, hidden_size=501, num_layers=3)
		self.fc2 = torch.nn.Linear(501, 76)
		if torch.cuda.is_available():
			self.cuda()

	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = x.repeat(277,1,1)
		x, _ = self.gru(x)
		x = self.fc2(x)
		x = x.permute(1,2,0)
		return x


class Vae(object):
	def __init__(self, batch_size):
		self.encoder = Encoder()
		self.decoder = Decoder()
		self.mask = Tensor(np.load("mask.npy"))
		self.train_dataset = Zinc_dataset(True)
		self.eval_dataset = Zinc_dataset(False)
		self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=True)
		self.eval_loader = DataLoader(dataset=self.eval_dataset, batch_size=batch_size, shuffle=False)

	def encode(self, x):
		return self.encoder(x)


	def decode(self, z):
		return self.decoder(z)
		
	def sample(self, mean, log_var):
		batch_size, latent_dim = mean.shape
		random = Tensor(np.random.normal(0, 1, size=(batch_size, latent_dim)))
		return mean + random * torch.exp(log_var / 2)
		
	def conditional_softmax(self, x_true, x_pred):
		x_pred = torch.exp(x_pred)
		x_pred_conditional = Tensor(np.zeros(shape=x_pred.shape))

		for i,(x_t,x_p) in enumerate(zip(x_true, x_pred)):
			rule_seq = torch.argmax(x_t, dim=0)
			mask = torch.index_select(self.mask, dim=1, index=rule_seq)
			divisor = torch.sum(torch.mul(x_p, mask), dim=0)
			x_pred_conditional[i] = x_p * mask / divisor
		return x_pred_conditional
		

	def loss(self, x):
		mean, log_var = self.encode(x)
		z = self.sample(mean, log_var)
		x_unmask = self.decoder(z)
		x_hat = self.conditional_softmax(x, x_unmask)
		reconstruct_loss = torch.sum(F.binary_cross_entropy(x_hat, x, reduction='none'), dim=(1,2))
		kl_loss = - 0.5 * torch.mean(1 + log_var - torch.pow(mean, 2) - torch.exp(log_var), dim=1)
		rule_seq = torch.argmax(x, dim=1)
		rule_seq_decode = torch.argmax(x_hat, dim=1)
		acc1 = torch.mean(torch.eq(rule_seq, rule_seq_decode).float(), dim=1)
		acc2 = (acc1 == 1).float()
		return torch.mean(reconstruct_loss + kl_loss), torch.mean(reconstruct_loss), torch.mean(kl_loss), torch.mean(acc1), torch.mean(acc2)

	def eval(self, loader, l):
		self.encoder.eval()
		self.decoder.eval()
		sum_all_loss = 0
		sum_recon_loss = 0
		sum_kl_loss = 0
		sum_acc1 = 0
		sum_acc2 = 0
		for x in loader:
			x = x.cuda()
			all_loss, recon_loss, kl_loss, acc1, acc2 = self.loss(x)
			sum_all_loss += all_loss.detach().cpu().numpy()*len(x)
			sum_recon_loss += recon_loss.detach().cpu().numpy()*len(x)
			sum_kl_loss += kl_loss.detach().cpu().numpy()*len(x)
			sum_acc1 += acc1.cpu().numpy()*len(x)
			sum_acc2 += acc2.cpu().numpy()*len(x)
		sum_all_loss /= l
		sum_recon_loss /= l
		sum_kl_loss /= l
		sum_acc1 /= l
		sum_acc2 /= l
		self.encoder.train()
		self.decoder.train()
		return sum_all_loss, sum_recon_loss, sum_kl_loss, sum_acc1, sum_acc2

	def train(self, epoch, doEval):
		logger = Logger("vaeInfo/log/")
		parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
		optimizer = torch.optim.Adam(parameters, lr=1e-5)
		# optimizer = torch.optim.SGD(parameters, lr=1e-5, momentum=0.9)
		best_acc = 0
		for e in range(epoch):
			sys.stdout.flush()
			print("Epoch %d" % e)
			for i,x in enumerate(self.train_loader):  
				x = x.cuda()			
				optimizer.zero_grad()
				batch_loss, recon_loss, kl_loss, acc1, acc2 = self.loss(x)
				batch_loss.backward()
				optimizer.step()
				# print("Batch No: %d, loss: %f, recon_loss: %f, kl_loss: %f, acc1: %f, acc2: %f" 
				# % (i+1, batch_loss, recon_loss, kl_loss, acc1, acc2))
				sys.stdout.flush()
			if doEval:
				all_loss, recon_loss, kl_loss, acc1, acc2 = self.eval(self.train_loader, len(self.train_dataset))
				print("Overall Training Loss: %f" % all_loss)
				print("Recontruction Training Loss: %f" % recon_loss)
				print("KL Training Loss: %f" % kl_loss)
				print("Rule Accuracy Training: %f" % acc1)
				print("Mol Accuracy Training: %f" % acc2)
				all_loss, recon_loss, kl_loss, acc1, acc2 = self.eval(self.eval_loader, len(self.eval_dataset))
				print("Overall Validation Loss: %f" % all_loss)
				print("Recontruction Validation Loss: %f" % recon_loss)
				print("KL Validation Loss: %f" % kl_loss)
				print("Rule Accuracy Validation: %f" % acc1)
				print("Mol Accuracy Validation: %f" % acc2)
				self.log_info(logger, all_loss, recon_loss, kl_loss, acc1, acc2)
				if acc2 > best_acc:
					print("Save Model")
					best_acc = acc2
					self.save()
					
	def sample_using_mask(self, unmask):
		prod_map = make_zinc_dataset_grammar.prod_map
		all_productions = zinc_grammar.GCFG.productions()
		start = all_productions[0].lhs()#smiles
		stack = [start]
		
		t = 0
		reconstruct_smile = ""
		while stack != []:
			current = stack.pop()			
			feasible = [prod_map[prod] for prod in prod_map if prod.lhs() == current]
			if feasible == []:#No left hand side matches, it is a terminal
				reconstruct_smile += str(current)
				continue
			all_mask = self.mask[feasible]
			for m in all_mask:
				assert torch.equal(m, all_mask[0])

			mask = all_mask[0].detach().cpu().numpy()
			x_t = unmask[0,:,t]
			masked_logit_exp = np.multiply(mask, np.exp(x_t))
			probability = masked_logit_exp / np.sum(masked_logit_exp) * (1 - 1e-6)
			# pick_lhs = np.argmax(np.random.multinomial(1, probability))
			pick_lhs = np.argmax(probability)
			for rhs in all_productions[pick_lhs].rhs()[::-1]:#push rhs into stack in a reversed order
				stack.append(rhs)
			if t == 276:
				break
			t += 1
		return reconstruct_smile
			
	def latent_view(self):
		self.encoder.eval()
		self.decoder.eval()
		# all_sample = []
		# for x in self.eval_loader:
		# 	x = x.cuda()
		# 	mean, log_var = self.encode(x)
		# 	z = self.sample(mean, log_var)
		# 	all_sample.append(z.detach().cpu().numpy())
		# all_sample = np.concatenate(all_sample)
		
		# u, _, _ = np.linalg.svd(np.matmul(all_sample.T, all_sample))
		# u_reduce = u[:,:2]
		# project_sample = np.matmul(all_sample, u_reduce)

		# plt.figure(figsize=(15,15))
		# x, y = zip(*project_sample)
		# plt.scatter(x, y, s=0.8)
		

		# plt.title("Vae Latent Space Distribution")
		# plt.axis("equal")
		# plt.savefig("VisVAEMol.png")
		# plt.close()

		one_hot = np.load("one_hot.npy")[:50]
		f = open('250k_rndm_zinc_drugs_clean.smi', 'r')
		L = []
		for line in f:
			line = line.strip()
			L.append(line)
		f.close()
		L = L[:50]
		cnt = 0
		for x,l in zip(one_hot,L):
			print("Encode: " + l)
			x = np.expand_dims(x, axis=0)
			x_hat = self.decode(self.sample(*self.encode(Tensor(x)))).detach().cpu().numpy()
			d = self.sample_using_mask(x_hat)
			print("Decode: " + d)
			if l == d:
				cnt += 1
		print(cnt/50)



		
		# all_x = np.arange(-6, -4.01, 0.2)
		# all_y = np.arange(-4, 4.01, 0.8)
		# xx, yy = np.meshgrid(all_x, all_y)
		# z = np.matmul(np.array(list(zip(xx.reshape(-1), yy.reshape(-1)))), u_reduce.T)
		# x_hat = self.decode(Tensor(z))
		# plt.figure(figsize=(150,150))
		# for i,x in enumerate(x_hat):
			# smile_dict = {}
			# for j in range(10000):
				# smile_recon = self.sample_using_mask(x.reshape(1,76,277))
				# m = Chem.MolFromSmiles(smile_recon)
				# try:
					# Draw.MolToImage(m)
					# if smile_recon in smile_dict:
						# smile_dict[smile_recon] += 1
					# else:
						# smile_dict[smile_recon] = 1
				# except:
					# continue
				# if j > 1000 and len(smile_dict) > 0:
					# break
			# if len(smile_dict) == 0:
				# print("10000!!!!!!!!!!")
				# exit(0)
			# else:
				# most_likely = sorted(smile_dict)[0]
				# m = Chem.MolFromSmiles(most_likely)
				# img = np.array(Draw.MolToImage(m))
				# w = i % 11 + 1
				# h = 10 - (i // 11)
				# plt.subplot(11, 11, h*11 + w)
				# plt.imshow(img)
				# plt.axis('off')
		# plt.savefig("TranAAEMol.png")
		# plt.close()
		
		
		
		self.encoder.train()
		self.decoder.train()


		

	def log_info(self, logger, all_loss, recon_loss, kl_loss, acc1, acc2):
		logger.log_value('Overall Loss', all_loss)
		logger.log_value('Recontruction Loss', recon_loss)
		logger.log_value('KL Loss', kl_loss)
		logger.log_value('Rule Acc', acc1)
		logger.log_value('Mol Acc', acc2)
		logger.step()


	def save(self):
		torch.save(self.encoder.state_dict(), "vaeInfo/model/encoder/" + "best.pth")
		torch.save(self.decoder.state_dict(), "vaeInfo/model/decoder/" + "best.pth")



	def load(self):
		self.encoder.load_state_dict(torch.load("vaeInfo/model/encoder/best.pth"))
		self.decoder.load_state_dict(torch.load("vaeInfo/model/decoder/best.pth"))



if __name__ == "__main__":
	# Load the dataset
	# get_gpu_memory_map()
	vae = Vae(100)
	# get_gpu_memory_map()
	vae.load()
	vae.latent_view()
	# vae.train(epoch=100, doEval=True)

	
	







	
	






