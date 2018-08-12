from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import glob
import time
import math
from Lang import Lang
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from AttDecoder import AttnDecoderRNN
from DecoderRnn import DecoderRNN
from EncoderRnn import EncoderRNN
import json
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import sklearn.model_selection._split

MAX_LENGTH = 50
SOS_token = torch.zeros(1947)
SOS_token[0] = 1

EOS_token = torch.zeros(1947)
EOS_token[1] = 1

teacher_forcing_ratio = 0.5
plt.switch_backend('agg')

capLang = None


def showPlot(points):
	plt.figure()
	fig, ax = plt.subplots()
	# this locator puts ticks at regular intervals
	loc = ticker.MultipleLocator(base=0.2)
	ax.yaxis.set_major_locator(loc)
	plt.plot(points)
	plt.savefig("experiment.png")
	

def asMinutes(s):
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)

#this function takes two arguments .
#"since" is the time when this training started.
#percent is  currentEpoch/total epoch yan kitna percent hogyahai
def timeSince(since, percent):
	now = time.time()
	s = now - since
	es = s / (percent)
	rs = es - s
	return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


#receive a sentence and return array of indexes of words in sentence (hotkey)
def indexesFromSentence(lang, sentence):
	return [lang.word2index[word] for word in sentence.split(' ')]

#append EOS and return Tensors
def tensorFromSentence(lang, sentence):
	indexes = indexesFromSentence(lang, sentence)
	indexes.append(EOS_token)
	#indexes = torch.tensor(indexes, dtype=torch.long)
	final = torch.tensor([])
	for i in indexes:
		if(len(final) == 0):
			final = torch.zeros(lang.n_words)
			final[torch.tensor(i)] = 1
			
		else:
			temp = torch.zeros(lang.n_words)
			temp[torch.tensor(i ,dtype=torch.long)] = 1
			final = torch.cat((final , temp) , dim=0)
	
	final = torch.cat((final , EOS_token))
	final = final.view(-1 , lang.n_words)
	return final


def tensorsFromPair(input_lang,pair):
	target_tensor = tensorFromSentence(input_lang, pair[1])
	
	input_tensor = torch.tensor(pair[0] , dtype=torch.float)
	return (input_tensor, target_tensor)

#remove non letter charecters
def normalizeString(s):
	s = re.sub(r"([.!?])", r" \1", s)
	s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
	return s

def getCaption(info , id):
	for v in info["sentences"]:
		if(v["video_id"] == "video" + str(id)):
			return normalizeString(v["caption"])


#Returns list of tuples
def GetfeatureVocabPair(file):
	print("reading ...")
	with open("info.json", 'r' , encoding='utf8') as f:
		info = json.load(f)
	arr = []
	list = glob.glob(file +"/*.npy")
	
	for l in list:
		
		feature = np.load(l.split("/")[-1])
		id = l.split("/")[-1].split(".npy")[0].split("video")[-1]
		caption = getCaption(info ,id)
		arr.append((feature , caption , id))
		print(id)
	return arr


#get pairs (videoFeature, caption) and build vocab
def PrepareData(file):
	pairs = GetfeatureVocabPair(file)
	capLang = Lang(name="EnglishCaptions")
	for pair in pairs:
		capLang.addSentence(pair[1])
	return capLang , pairs
	
	
#This function is single step in training
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
	encoder_hidden = encoder.initHidden("lstm")
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()
	
	#first dimention that is
	#input and output sequence lengths
	input_length = input_tensor.size(0)
	target_length = target_tensor.size(0)
	
	
	encoder_outputs = torch.zeros(max_length, encoder.hidden_size)
	
	loss = 0

	#input mein pehla frame dia sath hi hidden state di. phr us say output
	#or next hiden state ai wo phr say daidi. input sequence lenght tak yehi kia
	for ei in range(input_length):
		encoder_output, encoder_hidden = encoder (input_tensor[ei], encoder_hidden)
		encoder_outputs[ei] = encoder_output[0, 0]

	#deoder input mein abhi sirf SOS Token dala hai
	#pechay say jo hidden state aithi wo.
	
	decoder_input = torch.tensor(SOS_token)
	decoder_hidden = ((encoder_hidden[0][0] + encoder_hidden[0][1]).unsqueeze(0) , (encoder_hidden[1][0] + encoder_hidden[1][1]).unsqueeze(0))

	#use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
	use_teacher_forcing = False
	
	#print("train------------------------------------------------")
	if use_teacher_forcing:
		# Teacher forcing: Feed the target as the next input
		for di in range(target_length):
			decoder_output, decoder_hidden , decoder_att = decoder(decoder_input, decoder_hidden , encoder_outputs)
			target = torch.argmax(target_tensor[di]).view(1)
			predicted = torch.argmax(decoder_output.view(-1))
			#print("= " + str(capLang.index2word[int(target)]))
			#print("-> " + str(capLang.index2word[int(predicted)]))
			#print("loss = {}".format(loss))
			loss += criterion(torch.tensor(decoder_output.view(1 , -1) , dtype=torch.float) , target)
			decoder_input = target_tensor[di]  # Teacher forcing
			if (torch.argmax(decoder_input) == 1):
				break

	else:
		# Without teacher forcing: use its own predictions as the next input
		for di in range(target_length):
			decoder_output, decoder_hidden, decoder_att = decoder(decoder_input, decoder_hidden , encoder_outputs)
			decoder_input = decoder_output.view(-1).detach()
			target = torch.argmax(target_tensor[di]).view(1)
			predicted = predicted = torch.argmax(decoder_output.view(-1))
			#print("= " + str(capLang.index2word[int(target)]))
			#print("-> " + str(capLang.index2word[int(predicted)]))
			#print("loss = {}".format(loss))
			loss += criterion(torch.tensor(decoder_output.view(1 , -1) , dtype=torch.float), target)
			if (torch.argmax(decoder_input) == 1):
				break

	loss.backward()

	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.item() / target_length

	
def trainIters(language , pairs, encoder, decoder, epochs, print_every=1000, plot_every=50, learning_rate=0.01, cont=False ):
	if(cont):
		encoder.load_state_dict(torch.load("encoder.pt"))
		decoder.load_state_dict(torch.load("decoder.pt"))
		decoder = decoder.eval()
	
	start = time.time()
	plot_losses = []
	print_loss_total = 0  # Reset every print_every
	plot_loss_total = 0  # Reset every plot_every

	encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
	decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
	training_pairs = [tensorsFromPair(language, i) for i in pairs]
	criterion = nn.NLLLoss()

	#ye hai wo loop jis mien training ho rahi hai
	
	for i in range(1 , epochs+1):
		
		loss_total = np.array([])
		plot_loss_total  = np.array([])
		
		for iter in range(len(training_pairs)):
			training_pair = training_pairs[iter]
			input_tensor = training_pair[0]
			target_tensor = training_pair[1]
			
			#for one example
			loss = train(input_tensor, target_tensor, encoder,decoder, encoder_optimizer, decoder_optimizer, criterion)
			loss_total = np.append(loss_total, loss)
			plot_loss_total = np.append(plot_loss_total,loss)
	
		if i % print_every == 0:
			print("Epoch: {}\t loss = {}\nEst: {}\n".format(i, np.average(loss_total), timeSince(start , i/epochs)))
			
		if i % plot_every == 0:
			plot_loss_avg = np.average(plot_loss_total)
			plot_losses.append(plot_loss_avg)
			
		
	
	torch.save(encoder.state_dict() , "encoder.pt")
	torch.save(decoder.state_dict() , "decoder.pt")
	showPlot(plot_losses)


def evaluate(encoder, decoder, video, inputlang, max_length=MAX_LENGTH):
	encoder.load_state_dict(torch.load("encoder.pt"))
	decoder.load_state_dict(torch.load("decoder.pt"))
	decoder = decoder.eval()
	encoder_outputs = torch.zeros(max_length, encoder.hidden_size)
	encoder_hidden = encoder.initHidden("lstm")
	video = torch.tensor(video, dtype=torch.float)
	
	
	for ei in range(max_length):
		encoder_output, encoder_hidden = encoder (video[ei], encoder_hidden)
		encoder_outputs[ei] = encoder_output[0, 0]
	
	decoder_hidden = ((encoder_hidden[0][0] + encoder_hidden[0][1]).unsqueeze(0),
					  (encoder_hidden[1][0] + encoder_hidden[1][1]).unsqueeze(0))
	decoder_input = torch.tensor(SOS_token)
	Sentence = ""
	for di in range(max_length):
		word = inputlang.index2word[int(torch.argmax(decoder_input))]
		Sentence = Sentence + " " + word
		decoder_output, decoder_hidden , decoder_att = decoder(decoder_input, decoder_hidden , encoder_outputs)
		decoder_input = decoder_output.view(-1).detach()
		if (torch.argmax(decoder_input) == 1):
			break
	return Sentence


def GetCartoonPairs(arr, Pairs):
	newArr = []
	for i in arr:
		idvid = i.split("video")[-1]
		for p in Pairs:
			if(idvid == p[-1]):
				newArr.append(p)
				print(p[1])
				
	return newArr

#returns train , test
def trainTestSplit(arr , percentTrain):
	mid = int((percentTrain / 100) * len(arr))
	np.random.shuffle(arr)
	return arr[:mid] , arr[mid:]


def main():
	
	global capLang
	hidden_size = 128
	features_size = 128
	capLang , Pairs = PrepareData("featuresBig")
	TrainPairs , TestPairs = trainTestSplit(Pairs , 80)
	encoder1 = EncoderRNN(features_size  ,hidden_size , layers=1 )
	attn_decoder1 = AttnDecoderRNN(capLang.n_words , hidden_size,features_size, dropout_p=0.2)
	#decoder1 = DecoderRNN(capLang.n_words , hidden_size , layers=1)
	trainIters(capLang , TrainPairs , encoder1, attn_decoder1, epochs=120 , print_every=1, cont=False)
	
	
	cartoonID = open("cartoonID.txt", "r")
	arr = cartoonID.read().split(" ")
	Pairs = TestPairs
	#Pairs = GetCartoonPairs(arr , Pairs)
	#capLang, Pairs = PrepareData("featuresBig")
	results = open("results.txt","w")
	for i in range(len(Pairs)):
		pair = random.choice(Pairs)
		results.write("video" + str(pair[-1]) + str("\n"))
		results.write("Real:| " + str(pair[1])+str("\n"))
		print("video" + str(pair[-1]))
		print("Real:| " + str(pair[1]))
		sent = evaluate(encoder1, attn_decoder1, pair[0], capLang)
		results.write("predicted:| " + str(sent) + str("\n"))
		print("predicted:| " + str(sent) + str("\n\n"))
		
	results.close()
	

main()