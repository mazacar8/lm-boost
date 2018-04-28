# import gensim,logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import os 
import argparse
import time
# from neuralcoref.algorithm import Coref
# from pycorenlp import StanfordCoreNLP

# To do NLP processing to pick lines to swap:
# 1. Download neuralcoref module from https://github.com/huggingface/neuralcoref
# 2. Install requirements as needed (mentioned in the repo)
# 3. Uncomment import line above 
# 4. Initialize GenderPreProcess with controlled_swap = 1 for neuralcoref, 2 for Stanford deterministic coref

class MySentences(object):
	def __init__(self, dirname):
		self.dirname = dirname
 
	def __iter__(self):
		for fname in os.listdir(self.dirname):
			for line in open(os.path.join(self.dirname, fname)):
				yield line.split()

class GenderPreProcess:

	def __init__(self,en_file,gender_file,controlled_swap=0):

		self.en_file = en_file
		self.coref = Coref() if controlled_swap else None
		if controlled_swap == 1:
			self.coref = Coref()
			self.nlp = None
		elif controlled_swap == 2:
			self.nlp = StanfordCoreNLP('http://localhost:9000')
			self.coref = None
		else:
			self.coref = None
			self.nlp = None

		self.controlled_swap = controlled_swap
		self.num_ignored = 0
		self.total_gender = 0
		self.not_proper = 0
		self.corefTotalTime = 0.0
		with open(gender_file,'r') as f:
			lines = f.readlines()
			gp_lst = [line.strip().lower().split() for line in lines]
			self.male_search_words = [pair[0]+' ' for pair in gp_lst]
			self.female_search_words = [pair[1]+' ' for pair in gp_lst]
			gpd1= {pair[0]:pair[1] for pair in gp_lst}
			gpd2 = {pair[1]:pair[0] for pair in gp_lst}
			self.gender_pairs = {**gpd1,**gpd2}
		f.close()

	def __iter__(self):
		for fname in os.listdir(self.en_file):
			print('processing:',fname)
			i = 0
			for line in open(os.path.join(self.en_file, fname)):
				i += 1
				if i == 1000:
					print("Avg time for coref = %f"%(self.corefTotalTime/self.total_gender))
					break
				male = self.maleIndicated(line)
				female = self.femaleIndicated(line)
				genderIndicated = male or female
				self.total_gender += 1 if genderIndicated else 0
				if genderIndicated and self.shouldSwap(line):
					 swapped_line = self.swapGender(line)
					 yield swapped_line
#                if genderIndicated:
#                    yield line

	#Simple proper noun check. Only checks for capitalization and "A"/"The"
	def isProper(self,s):

		words = s.split(' ')
		first = words[0]
		checks = ['The','A']
		if first.islower() or first in checks:
			return False
		return True 


	def huggingCoref(self,line):

		clusters = self.coref.one_shot_coref(line)
		references = self.coref.get_most_representative()
		gender_ref = False

		for ref in references:
			#If a gender indicative word has a stong coreference resolution
			#with another part of a sentence. Should ideally change it so 
			#it checks for a strong coreference with a proper noun
			if str(ref) in self.gender_pairs:
				# print(references)
				gender_ref = True
				val = references[ref]
				#Don't swap if gender indicator is related to a proper noun
				if self.isProper(str(val)):
					self.num_ignored += 1
					print(line)
					return False

		if gender_ref:
			self.not_proper += 1

		return True

	def stanfordCoref(self,line):
		output = self.nlp.annotate(line, properties={'annotators': 'dcoref','outputFormat': 'json'})
		print(line)
		references = output['corefs']
		proper = False
		gender = False
		for ref_id in references:
			mentions = references[ref_id]
			for mention in mentions:
				if mention['gender'] == 'MALE' or mention['gender'] == 'FEMALE':
					gender = True
				if mention['type'] == 'PROPER':
					proper = True

				if gender and proper:
					# print(line)
					return False

			gender = False
			proper = False

		return True

	#Checks if gender indicators are there for a reason
	#Example: "George Bush said that he is taking action" should not be swapped
	def shouldSwap(self,line):

		startTime = time.time()
		if self.controlled_swap == 0:
			self.corefTotalTime += (time.time()-startTime)
			return True

		elif self.controlled_swap == 1:
			res = self.huggingCoref(line)
			self.corefTotalTime += (time.time()-startTime)
			return res

		elif self.controlled_swap == 2:
			res = self.stanfordCoref(line)
			self.corefTotalTime += (time.time()-startTime)
			return res

	#Determines if a given sentence has any indication of the male gender
	def maleIndicated(self,line):
		lc_line = line.strip().lower()
		for w in self.male_search_words:
			spaced = " "+w
			if (lc_line[0:len(w)] == w) or (spaced in lc_line):
				return True
		return False

	#Determines if a given sentence has any indication of the female gender
	def femaleIndicated(self,line):
		lc_line = line.strip().lower()
		for w in self.female_search_words:
			spaced = " "+w
			if (lc_line[0:len(w)] == w) or (spaced in lc_line):
				return True
		return False

	#Swaps all gender specific words in a given sentence
	#Inspired by https://www.geeksforgeeks.org/change-gender-given-string/
	def swapGender(self,line):
		words = line.strip().split(' ')
		case = [1 if ((word[0].isupper()) or (word[0] == "\"" \
			and len(word) > 1 and word[1].isupper())) else 0 for word in words]
		words = [word.lower() for word in words]
		out_words = [self.gender_pairs[word] if word in self.gender_pairs \
		else word for word in words]
		out_words = [word.title() if case[i] else word for (i,word) in \
		enumerate(out_words)]
		out_sentence = ' '.join(word for word in out_words)

		return out_sentence+'\n'
		
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=\
			"Train an Embedding")
	parser.add_argument('data_folder', metavar='infile1', type=str, \
		help='folder containing the data')
	
	parser.add_argument('gender_pair_path', metavar='infile2', type=str, \
	help='file for gender pairs')
	
	parser.add_argument('out_path', metavar='outfile1', type=str, \
	help='out path for swapped files')
	args = parser.parse_args()
	sentences = GenderPreProcess(args.data_folder,args.gender_pair_path,2) # a memory-friendly iterator
	f = open(args.out_path,'w')
	start = time.time()
	for s in sentences:
		f.write(s)
	f.close()
	print("Total Time = %f"%(time.time()-start))
