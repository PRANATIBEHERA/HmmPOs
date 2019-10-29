from sklearn.model_selection import train_test_split
train_data = open('Brown_train.txt','r')
train_data_line = train_data.readlines()
train, test = train_test_split(train_data_line, train_size = 0.7,test_size = 0.3)
result_file = open('result_file_pos.txt','a')
result_tag_file=open('result_tag_file.txt','a')
result_confusion_matrix=open('result_confusion_matrix.txt','a')

tag_pair_count = dict()
tag_pair_prob = dict()
tag_count = dict()
transition = dict()
emission =dict()
word_tag_pair=dict()
start=dict()
confusion_matrix = dict()
confusion_matrix_final = dict()
total_word=0

def trans_count(train):
	for line in train:
		list_ = []
		line = line.strip("\n")
		words = line.split(" ")
		for i in range(0,len(words)-1):
			i_split = words[i].split("/")
			list_.append(i_split[1])
		list_.append("^")
		for j in range(0,len(list_)-1):
			if(list_[j],list_[j+1]) not in tag_pair_count:
				tag_pair_count[list_[j],list_[j+1]]=1
			else:
				tag_pair_count[list_[j],list_[j+1]]=tag_pair_count[list_[j],list_[j+1]]+1
	return tag_pair_count
# print tag_pair_count

""" divide tag pair with the probablity of the tag """
def trans_prob(lexicon):
	for key, value in lexicon.iteritems():
		if key[0] not in tag_count:
			tag_count[key[0]] = value 
		else:
			tag_count[key[0]] = tag_count[key[0]]+value
	for key,value in lexicon.iteritems():
		if key not in tag_pair_prob:
			tag_pair_prob[key]=float(value)/float((tag_count[key[0]]))
	return tag_pair_prob
""" it is used to find the unique tag and unique vocabulary so that nothing is hardcoded"""
def unique_tag(train):
	states = []
	for line in train:
		line = line.strip("\n")
		words = line.split(" ")
		for i in range(0,len(words)-1):
			i_split = words[i].split("/")
			if i_split[1] not in states:
				states.append(i_split[1])
	return states


def missing_prob(lexicon,list1,list2):
	for i in range(len(list1)):
		for j in range(len(list2)):
			if (list1[i],list2[j]) not in lexicon:
				transition[list1[i],list2[j]]=float(0.000001)
			else:
				transition[list1[i],list2[j]]=float(lexicon[list1[i],list2[j]])
	return transition

def calculate_transition():
	trans_count(train)
	trans_prob(tag_pair_count)
	states=unique_tag(train_data_line)
	transition=missing_prob(tag_pair_prob,states,states)
	# print transition


def word_tag_count(train):
	for line in train:
		line = line.strip("\n")
		words = line.split(" ")
		for i in range(0,len(words)-1):
			i_split = words[i].split("/")
			if(i_split[0],i_split[1]) not in word_tag_pair:
				word_tag_pair[i_split[0],i_split[1]]=1
			else:
				word_tag_pair[i_split[0],i_split[1]]=word_tag_pair[i_split[0],i_split[1]]+1
	return word_tag_pair

def word_tag_prob(lexicon):
	for key,value in lexicon.iteritems():
		word_tag_pair[key]=float(value)/float(tag_count[key[1]])
	return word_tag_pair
def unique_vocab(lexicon):
	list_=[]
	for key,value in lexicon.iteritems():
		if key[0] not in list_:
			list_.append(key[0])
	return list_


def emission_table():
	list2=unique_tag(train_data_line)
	list1=unique_vocab(word_tag_pair)
	for i in range(len(list1)):
		for j in range(len(list2)):
			if(list1[i],list2[j]) not in word_tag_pair:
				emission[list1[i],list2[j]]=float(0.00001)
			else:
				emission[list1[i],list2[j]]=word_tag_pair[list1[i],list2[j]]
	return emission
def calculate_emiss_probablity():
	word_tag_count(train)
	word_tag_prob(word_tag_pair)
	emission_table()
	# print emission
states=unique_tag(train_data_line)
# print states

def calculate_start_prob(train,states):
	count =0
	for line in train:
		count = count+1
		line = line.strip("\n")
		words = line.split(" ")
		i_split = words[0].split("/")
		if (i_split[1]) not in start:
			start[i_split[1]]=1
		else:
			start[i_split[1]]=start[i_split[1]]+1
	for key,value in start.iteritems():
		start[key]=float(value)/float(count)
	for i in range(len(states)):
		if (states[i]) not in start:
			start[states[i]]=0.000001
	return start
def round_prob_5_digit(transition,emission):
	for key,value in transition.iteritems():
		transition[key]=round(value,7)
	for key,value in emission.iteritems():
		emission[key] = round(value,7)
	return transition,emission

calculate_transition()
calculate_emiss_probablity()
round_prob_5_digit(transition,emission)
calculate_start_prob(train,states)
# print start
def write_file(wordlist,taglist,opt):
	for i in range(len(wordlist)):
		string=wordlist[i]+"\t"+taglist[i]+"\t"+opt[i]
		#print string
		result_file.write(string)
		result_file.write("\n")
def seperate_tag_word(line):
	wordlist=[]
	taglist=[]
	line = line.strip("\n")
	words = line.split(" ")
	for i in range(0,len(words)-1):
			i_split = words[i].split("/")
			wordlist.append(i_split[0])
			taglist.append(i_split[1])
	return wordlist,taglist
def emiss_prob(emission,word,tag):
	
	if(word,tag) not in emission:
		return (float(0.00001))
	else:
		if(emission[word,tag])>float(0.0):
			return round(emission[word,tag],7)
		else:
			return float(0.00001)


def create_confusion_matrix(taglist,opt):
	for i in range(0,len(taglist)):
		if(taglist[i],opt[i]) not in confusion_matrix:
			confusion_matrix[taglist[i],opt[i]]=1
		else:
			confusion_matrix[taglist[i],opt[i]]=confusion_matrix[taglist[i],opt[i]]+1
	# for i in range(0,len(states)):
	# 	for j in range(0,len(states)):
	# 		if (states[i],states[j]) not in confusion_matrix:
	# 			confusion_matrix[states[i],states[j]]=int(0)

	

def calculate_accuracy():
	result=0
	precision=0
	recall=0
	accuracy=0
	f1_score=0
	for key,value in confusion_matrix.iteritems():
		if(key[0]==key[1]):
			result=result+confusion_matrix[key]
	accuracy=float(result)/float(total_word)
	precision=accuracy
	recall=accuracy
	f1_score=accuracy
	return accuracy,precision,recall,f1_score
precision_tag =dict()
recall_tag=dict()
f1_score_tag = dict()
def calculate_num(tag):
	if (tag,tag) in confusion_matrix:
		return confusion_matrix[tag,tag]
	else:
		return 0
def calculate_denom(tag,index):
	value_tag=0
	for key,value in confusion_matrix.iteritems():
		if(tag==key[index]):
			value_tag=value_tag+value
	if(value_tag == 0):
		value_tag=1
	return value_tag		



def calculate_tag_wise():
	#for precision
	for st in states:
		num_precision=calculate_num(st)
		denom_precision=calculate_denom(st,1)
		if(denom_precision)==0:
			precision_tag[st]=float(0.00)
		else:
			precision_tag[st]=float(num_precision)/float(denom_precision)
	#recall
	for st in states:
		num_recall=calculate_num(st)
		denom_recall=calculate_denom(st,0)
		if(denom_recall)==0:
			recall_tag[st]=float(0.00)
		else:
			recall_tag[st]=float(num_recall)/float(denom_recall)
	# f1 score =2pr/p+r
	for st in states:
		num_f1 = 2 * precision_tag[st] * recall_tag[st]
		denom_f1 = precision_tag[st] + recall_tag[st]
		if(denom_f1 == 0):
			f1_score_tag[st]=float(0.00)
		else:
			f1_score_tag[st]=float(num_f1)/float(denom_f1)

def print_precision_recall_f1_accuracy():
	for st in states:
		string=""
		string="tag:"+st+"\t "+"precision:"+str(round(precision_tag[st],2))+"\t "+"recall:"+str(round(recall_tag[st],2))+"\t "+"f1-score:"+str(round(f1_score_tag[st],2))
		print string
		result_tag_file.write(string)
		result_tag_file.write("\n")
def print_confusion_matrix():
	count =0
	for st in states:
		for st1 in states:
			string =""
			if(st,st1) not in confusion_matrix:
				result=0
			else:
				result=confusion_matrix[st,st1]
				count =count+1
			string="actual tag:"+" "+st+"\t"+"predicted tag:"+" "+st1+"\t"+str(result)
			print string
			result_confusion_matrix.write(string)
			result_confusion_matrix.write("\n")

	print count	





def viterbi(test,states,emission,transition,start,total_word):
	# print transition
	V = [{}]
	count =0
	total_count=0
	for line in test:
		count = count +1
		# print count
		V = [{}]
		wordlist,taglist=seperate_tag_word(line)
		total_count = total_count + len(wordlist)
		# print total_word
		# print type(total_word)
		for st in states:
			value =float(emiss_prob(emission,wordlist[0],st))
			V[0][st] = {"prob": start[st] * value, "prev": None}
		for t in range(1, len(wordlist)):
			V.append({})
			for st in states:
				max_tr_prob = max(V[t-1][prev_st]["prob"]*transition[prev_st,st] for prev_st in states)
				for prev_st in states:
					if V[t-1][prev_st]["prob"] * transition[prev_st,st] == max_tr_prob:
						value =float(emiss_prob(emission,wordlist[t],st))
						max_prob = max_tr_prob * value
						V[t][st] = {"prob": max_prob, "prev": prev_st}
						break

		opt = []
		max_prob = max(value["prob"] for value in V[-1].values())
		previous = None
		# print max_prob
		for st, data in V[-1].items():
			if data["prob"] == max_prob:
				opt.append(st)
				previous = st
				break
		for t in range(len(V) - 2, -1, -1):
			opt.insert(0, V[t + 1][previous]["prev"])
			previous = V[t + 1][previous]["prev"]
		write_file(wordlist,taglist,opt)
		#print wordlist
		#print count
		create_confusion_matrix(taglist,opt)
	total_word=total_count
	return total_word

total_word=viterbi(test,states,emission,transition,start,total_word)
accuracy,precision,recall,f1_score=calculate_accuracy()
print("overall acurracy"+" "+str(accuracy))

print("overall precision"+" "+str(precision))

print("overall recall"+" "+str(recall))
print("overall f1_score"+" "+str(f1_score))

calculate_tag_wise()
#print_confusion_matrix()
print_precision_recall_f1_accuracy()
