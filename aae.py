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
		
		
class Discriminator(torch.nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.fc1 = torch.nn.Linear(latent_rep_size, 256)
		self.fc2 = torch.nn.Linear(256, 256)
		self.fc3 = torch.nn.Linear(256, 1)
		if torch.cuda.is_available():
			self.cuda()
			
	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		x = self.fc3(x)
		return x


class Aae(object):
	def __init__(self, batch_size):
		self.encoder = Encoder()
		self.decoder = Decoder()
		self.discriminator = Discriminator()
		self.mask = Tensor(np.load("mask.npy"))
		self.train_dataset = Zinc_dataset(True)
		self.eval_dataset = Zinc_dataset(False)
		self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=True)
		self.eval_loader = DataLoader(dataset=self.eval_dataset, batch_size=batch_size, shuffle=False)


	def encode(self, x):
		return self.encoder(x)


	def decode(self, z):
		return self.decoder(z)

	def discri(self, z):
		return self.discriminator(z)

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
		


	def reconstruct_loss(self, x):
		mean, log_var = self.encode(x)
		z = self.sample(mean, log_var)
		x_unmask = self.decoder(z)
		x_hat = self.conditional_softmax(x, x_unmask)
		loss = torch.sum(F.binary_cross_entropy(x_hat, x, reduction='none'), dim=(1,2))
		rule_seq = torch.argmax(x, dim=1)
		rule_seq_decode = torch.argmax(x_hat, dim=1)
		acc1 = torch.mean(torch.eq(rule_seq, rule_seq_decode).float(), dim=1)
		acc2 = (acc1 == 1).float()
		return torch.mean(loss), torch.mean(acc1), torch.mean(acc2)
	
	def dicriminate_loss(self, x):
		mean_fake, log_var_fake = self.encode(x)
		z_fake = self.sample(mean_fake, log_var_fake)

		batch_size, latent_dim = z_fake.shape

		mean_true = Tensor(np.zeros(shape=(batch_size, latent_dim)))
		log_var_ture = Tensor(np.zeros(shape=(batch_size, latent_dim)))
		z_true = self.sample(mean_true, log_var_ture)

		all_z = torch.cat((z_fake, z_true), dim=0)
		logit = self.discri(all_z).reshape(-1)
		
		label_fake = Tensor(np.zeros(shape=(batch_size)))
		label_true = Tensor(np.ones(shape=(batch_size)))
		all_label = torch.cat((label_fake, label_true))
		
		pred_label = (logit > 0.5).float()
		acc = torch.mean((pred_label == all_label).float())

		loss = F.binary_cross_entropy_with_logits(logit, all_label)
		return loss, acc
		
	def fool_loss(self, x):
		mean, log_var = self.encode(x)
		z_fake = self.sample(mean, log_var)
		logit = self.discri(z_fake).reshape(-1)
		label_true = Tensor(np.ones(shape=(mean.shape[0])))
		pred_label = (logit > 0.5).float()
		acc = torch.mean((pred_label == label_true).float())
		loss = F.binary_cross_entropy_with_logits(logit, label_true)
		return loss, acc
	
	def eval(self, loader, l):
		self.encoder.eval()
		self.decoder.eval()
		self.discriminator.eval()

		sum_recon_loss = 0
		sum_discr_loss = 0
		sum_fool_loss = 0
		sum_recon_acc1 = 0
		sum_recon_acc2 = 0
		sum_discr_acc = 0
		sum_fool_acc = 0
		for x in loader:
			x = x.cuda()
			recon_loss, recon_acc1, recon_acc2 = self.reconstruct_loss(x)
			discr_loss, discr_acc = self.dicriminate_loss(x)
			fool_loss, fool_acc = self.fool_loss(x)
			sum_recon_loss += recon_loss.detach().cpu().numpy()*len(x)
			sum_recon_acc1 += recon_acc1.detach().cpu().numpy()*len(x)
			sum_recon_acc2 += recon_acc2.detach().cpu().numpy()*len(x)
			sum_discr_loss += discr_loss.detach().cpu().numpy()*len(x)
			sum_discr_acc += discr_acc.detach().cpu().numpy()*len(x)
			sum_fool_loss += fool_loss.detach().cpu().numpy()*len(x)
			sum_fool_acc += fool_acc.detach().cpu().numpy()*len(x)
		sum_recon_loss /= l
		sum_recon_acc1 /= l
		sum_recon_acc2 /= l
		sum_discr_loss /= l
		sum_discr_acc /= l
		sum_fool_acc /= l
		sum_fool_loss /= l
		self.encoder.train()
		self.decoder.train()
		self.discriminator.train()
		return sum_recon_loss, sum_recon_acc1, sum_recon_acc2, sum_discr_loss, sum_discr_acc, sum_fool_loss, sum_fool_acc

	def train(self, epoch, doEval):
		logger = Logger("aaeInfo/log/")
		parameters_recon = list(self.encoder.parameters()) + list(self.decoder.parameters())
		optimizer_recon = torch.optim.SGD(parameters_recon, lr=1e-6, momentum=0.9)
		optimizer_discr = torch.optim.SGD(self.discriminator.parameters(), lr=1e-6, momentum=0.1)
		optimizer_gener = torch.optim.SGD(self.encoder.parameters(), lr=1e-6, momentum=0.1)
		# optimizer_recon = torch.optim.Adam(parameters_recon, betas=(0.9, 0.999), lr=1e-5)
		# optimizer_discr = torch.optim.Adam(self.discriminator.parameters(), betas=(0.1, 0.001), lr=1e-4)
		# optimizer_gener = torch.optim.Adam(self.encoder.parameters(), betas=(0.1, 0.001), lr=1e-4)

		best_loss = float("Inf")
		for e in range(epoch):
			sys.stdout.flush()
			print("Epoch %d" % e)
			for i,x in enumerate(self.train_loader):
				x = x.cuda()
				optimizer_recon.zero_grad()
				recon_loss, recon_acc1, recon_acc2 = self.reconstruct_loss(x)
				recon_loss.backward()
				optimizer_recon.step()
				
				
				optimizer_discr.zero_grad()
				discr_loss, discr_acc = self.dicriminate_loss(x)
				discr_loss.backward()
				optimizer_discr.step()
				
				optimizer_gener.zero_grad()
				fool_loss, fool_acc = self.fool_loss(x)
				fool_loss.backward()
				optimizer_gener.step()
				
				print("Batch No: %d, recon_loss: %f, recon_acc1: %f, recon_acc2: %f, discr_loss: %f, discr_acc: %f, fool_loss: %f, fool_acc: %f" 
				% (i+1, recon_loss, recon_acc1, recon_acc2, discr_loss, discr_acc, fool_loss, fool_acc))
				sys.stdout.flush()
				
				
			if doEval:
				recon_loss, recon_acc1, recon_acc2, discri_loss, discr_acc, fool_loss, fool_acc = self.eval(self.train_loader, len(self.train_dataset))
				print("Reconstruction Train Loss: %f, Rule Accuracy: %f, Mol Accuracy: %f" % (recon_loss, recon_acc1, recon_acc2))
				print("Discrimination Train Loss: %f, Accuracy: %f" % (discri_loss, discr_acc))
				print("Fool Train Loss: %f, Accuracy: %f" % (fool_loss, fool_acc))
				recon_loss, recon_acc1, recon_acc2, discri_loss, discr_acc, fool_loss, fool_acc = self.eval(self.eval_loader, len(self.eval_dataset))
				print("Reconstruction Valid Loss: %f, Rule Accuracy: %f, Mol Accuracy: %f" % (recon_loss, recon_acc1, recon_acc2))
				print("Discrimination Valid Loss: %f, Accuracy: %f" % (discri_loss, discr_acc))
				print("Fool Valid Loss: %f, Accuracy: %f" % (fool_loss, fool_acc))
				self.log_info(logger, recon_loss, recon_acc1, recon_acc2, discri_loss, discr_acc, fool_loss,  fool_acc)
				if recon_loss < best_loss:
					print("Save Model")
					best_loss = recon_loss
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
			mask = all_mask[0].detach().numpy()
			x_t = unmask[0,:,t].detach().numpy()
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
		self.discriminator.eval()
		all_sample = []
		for x in self.eval_loader:
			x = x.cuda()
			mean, log_var = self.encode(x)
			z = self.sample(mean, log_var)
			all_sample.append(z.detach().cpu().numpy())
		all_sample = np.concatenate(all_sample)
		
		u, _, _ = np.linalg.svd(np.matmul(all_sample.T, all_sample))
		u_reduce = u[:,:2]
		project_sample = np.matmul(all_sample, u_reduce)

		plt.figure(figsize=(15,15))
		x, y = zip(*project_sample)
		plt.scatter(x, y, s=0.8)
		# plt.plot([-6,-4],[-4,-4], color="k")
		# plt.plot([-6,-4],[4,4], color="k")
		# plt.plot([-6,-6],[-4,4], color="k")
		# plt.plot([-4,-4],[-4,4], color="k")
		

		plt.title("Aae Latent Space Distribution")
		plt.axis("equal")
		plt.savefig("VisAAEMol.png")
		plt.close()
		
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
		self.discriminator.train()
			
		



	def log_info(self, logger, recon_loss, recon_acc1, recon_acc2, discri_loss, discr_acc, fool_loss, fool_acc):
		logger.log_value('Reconstruction Loss', recon_loss)
		logger.log_value('Reconstruction Rule Accuracy', recon_acc1)
		logger.log_value('Reconstruction Mol Accuracy', recon_acc2)
		logger.log_value('Discrimination Loss', discri_loss)
		logger.log_value('Discrimination Accuracy', discr_acc)
		logger.log_value('Fool Loss', fool_loss)
		logger.log_value('Fool Accuracy', fool_acc)
		logger.step()


	def save(self):
		torch.save(self.encoder.state_dict(), "aaeInfo/model/encoder/" + "best.pth")
		torch.save(self.decoder.state_dict(), "aaeInfo/model/decoder/" + "best.pth")
		torch.save(self.discriminator.state_dict(), "aaeInfo/model/discriminator/" + "best.pth")


	def load(self):
		self.encoder.load_state_dict(torch.load("aaeInfo/model/encoder/best.pth"))
		self.decoder.load_state_dict(torch.load("aaeInfo/model/decoder/best.pth"))
		self.discriminator.load_state_dict(torch.load("aaeInfo/model/discriminator/best.pth"))



if __name__ == "__main__":
	# Load the dataset
	
	get_gpu_memory_map()
	aae = Aae(200)
	get_gpu_memory_map()
	aae.load()
	aae.latent_view()
	aae.train(epoch=100, doEval=True)
	



	
	






