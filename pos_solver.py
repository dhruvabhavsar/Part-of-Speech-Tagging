###################################
# CS B551 Fall 2019, Assignment #3
#
# Your names and user ids: Dhruva Bhavsar(dbhavsar); Hely Modi(helymodi); Aneri Shah(annishah)
#
# (Based on skeleton code by D. Crandall)
#


import random
import math
import numpy as np

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        posterior_log=[]
        
        
        if model == "Simple":
            for i in range(len(sentence)):
                if i==0:
                    if sentence[i] in self.emis_prob:
                        posterior_log.append(math.log(self.emis_prob[sentence[i]][label[i]]*self.prob_first[label[i]]))            
                    else:
                        posterior_log.append(math.log(1/100000*self.prob_first[label[i]]))   
                else:
                    if sentence[i] in self.emis_prob:
                        posterior_log.append(math.log(self.emis_prob[sentence[i]][label[i]]*self.trans_prob[label[i-1]][label[i]]))            
                    else:
                        posterior_log.append(math.log(1/100000*self.trans_prob[label[i-1]][label[i]]))  
            return sum(posterior_log)
        elif model == "Complex":
            for i in range(len(sentence)):
                if i==0:
                    if sentence[i] in self.emis_prob:
                        posterior_log.append(math.log(self.emis_prob[sentence[i]][label[i]]*self.prob_first[label[i]]))            
                    else:
                        posterior_log.append(math.log(1/100000*self.prob_first[label[i]]))   
                elif i==len(sentence)-1:
                    if sentence[i] in self.emis_prob:
                        posterior_log.append(math.log(self.emis_prob[sentence[i]][label[i]]*self.trans_prob[label[i-1]][label[i]]*self.first_last_prob[label[i]][label[0]]))            
                    else:
                        posterior_log.append(math.log(1/100000*self.trans_prob[label[i-1]][label[i]]*self.first_last_prob[label[i]][label[0]])) 
                else:
                    if sentence[i] in self.emis_prob:
                        posterior_log.append(math.log(self.emis_prob[sentence[i]][label[i]]*self.trans_prob[label[i-1]][label[i]]))            
                    else:
                        posterior_log.append(math.log(1/100000*self.trans_prob[label[i-1]][label[i]]))  
            return sum(posterior_log)
        elif model == "HMM":
            for i in range(len(sentence)):
                if i==0:
                    if sentence[i] in self.emis_prob:
                        posterior_log.append(math.log(self.emis_prob[sentence[i]][label[i]]*self.prob_first[label[i]]))            
                    else:
                        posterior_log.append(math.log(1/100000*self.prob_first[label[i]]))   
                else:
                    if sentence[i] in self.emis_prob:
                        posterior_log.append(math.log(self.emis_prob[sentence[i]][label[i]]*self.trans_prob[label[i-1]][label[i]]))            
                    else:
                        posterior_log.append(math.log(1/100000*self.trans_prob[label[i-1]][label[i]]))  
            return sum(posterior_log)
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):
        self.words={}
        self.emis_prob={}
        self.trans_prob={}
        
        s1=''
        count=1/1000000
        self.first_last_prob={}
        self.prob_first={'adj':count,'adv':count,'adp':count,'conj':count,'det':count,'noun':count,'num':count,'pron':count,'prt':count,'verb':count,'x':count,'.':count}
        self.prob_last={'adj':count,'adv':count,'adp':count,'conj':count,'det':count,'noun':count,'num':count,'pron':count,'prt':count,'verb':count,'x':count,'.':count}
        total_count_last = 0
        total_count_first = 0
        for sen in data:
            sen1,sen2=sen
            
            index=0
            for w,s in zip(sen1,sen2):

                if index == 0:
                    self.prob_first[s]+=1
                    total_count_first += 1
                    fw=s
                elif index == len(sen[1]) - 1:
                    self.prob_last[s]+=1
                    total_count_last += 1
                    lw=s
                
                # Count for emission probability
                if w not in self.words:
                    self.words[w]={'adj':count,'adv':count,'adp':count,'conj':count,'det':count,'noun':count,'num':count,'pron':count,'prt':count,'verb':count,'x':count,'.':count}
                self.words[w][s]+=1
                
                # Count for transition probability
                if(s1!=''):
                    if s1 not in self.trans_prob:
                        self.trans_prob[s1]={'adj':count,'adv':count,'adp':count,'conj':count,'det':count,'noun':count,'num':count,'pron':count,'prt':count,'verb':count,'x':count,'.':count}
                        self.trans_prob[s1][s]=1
                    else:
                        self.trans_prob[s1][s]+=1
                s1=s
                index+=1
            # Calculate probability of last POS given first POS
            if fw not in self.first_last_prob:
                self.first_last_prob[fw]={'adj':count,'adv':count,'adp':count,'conj':count,'det':count,'noun':count,'num':count,'pron':count,'prt':count,'verb':count,'x':count,'.':count}
                self.first_last_prob[fw][lw]=1
            else:
                self.first_last_prob[fw][lw]+=1
                
        for s1 in self.first_last_prob:
            total=sum(self.trans_prob[s1].values())
            for s in self.first_last_prob[s1]:
                self.first_last_prob[s1][s]=self.first_last_prob[s1][s]/total
        
        # Calculate prior probabilities
        for s1 in self.prob_first:
                self.prob_first[s1]=self.prob_first[s1]/total_count_first

        # Final transition probabilities:
        for s1 in self.trans_prob:
            total=sum(self.trans_prob[s1].values())
            for s in self.trans_prob[s1]:
                self.trans_prob[s1][s]=self.trans_prob[s1][s]/total

     # Probability of each type of POS
        total_prob={}
        for w in self.words:
            for s in self.words[w]:
                if s not in total_prob:
                    total_prob[s]=self.words[w][s]
                else:
                    total_prob[s]+=self.words[w][s]
        total=sum(total_prob.values())
        
        # Final emission probabilities:
        for w in self.words:
            self.emis_prob[w]={'adj':count,'adv':count,'adp':count,'conj':count,'det':count,'noun':count,'num':count,'pron':count,'prt':count,'verb':count,'x':count,'.':count}
            for s in self.words[w]:
                self.emis_prob[w][s]=self.words[w][s]/(total_prob[s]*sum(self.words[w].values()))

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        # Simplifies approach which makes use of only emission probabilities
        s=[]
        for w in sentence:
            if w in self.words:
                s.append(max(self.words[w], key=self.words[w].get))
            else:
                s.append("noun")
        return s
    
    # Gibb's sampling:
    def complex_mcmc(self, sentence):
        # Initial sample
        sample = ["noun"] * len(sentence)
        samples=[]
        samples.append(sample)
        initial=samples[0]
        types = {0:'adj',1:'adv',2:'adp',3:'conj',4:'det',5:'noun',6:'num',7:'pron',8:'prt',9:'verb',10:'x',11:'.'}
        
        # Run for 5000 iterations:
        for i in range(5000):
            for j in range(len(sentence)):
                prob=[]
                prob_sum=0
                for k in range(12):
                    new=types[k]
                    a=0
                    b=1
                    e_p=1/1000000
                    
                    # For first POS tag:
                    if j==0:
                        a=self.prob_first[new]
                    # For last POS tag:
                    elif j==len(sentence)-1:
                        a=self.trans_prob[initial[j-1]][new]*self.first_last_prob[initial[0]][new]
                    else:
                        a=self.trans_prob[initial[j-1]][new]
                        
                    if sentence[j] in self.emis_prob:
                        e_p=self.emis_prob[sentence[j]][new]
                    
                    if k<len(sentence)-1:
                        b=self.trans_prob[initial[k+1]][new]
                        
                    temp_val=a*b*e_p
                    
                    prob_sum+=temp_val
                    
                    prob.append(temp_val)
                    
                c=0
                r=random.random()

                for q in range(len(prob)):
                    prob[q]=prob[q]/prob_sum
                    c+=prob[q]
                    prob[q]=c
                    if r<prob[q]:
                        o=q
                        break
                
                initial[j]=types[o]
            samples.append(initial)
        
        top_value=samples[len(samples)-1:]            

        return(top_value[0])
     
    # Viterbi implementation:
    def hmm_viterbi(self, sentence):

        S = len(sentence)
        T = len(self.trans_prob)

        V = np.zeros((S, T))
        count=1/1000000
        prev = np.zeros((S-1,T))

        types = {0:'adj',1:'adv',2:'adp',3:'conj',4:'det',5:'noun',6:'num',7:'pron',8:'prt',9:'verb',10:'x',11:'.'}

        for t in range(0,len(sentence)):

            word=sentence[t]

            for j in range(0,len(types)):
                if t==0:
                    if word in self.emis_prob:
                        V[t][j]=np.log(self.prob_first[types[j]])+np.log(self.emis_prob[word][types[j]])
                    else:
                        V[t][j]=np.log(self.prob_first[types[j]]*count)
                else:
                    
                    prob = [V[t - 1][i] + np.log(self.trans_prob[types[i]][types[j]]) for i in range(12)]

                    prev[t-1][j]= np.argmax(prob)
                    if word in self.emis_prob:
                        V[t][j] = max(prob)+np.log(self.emis_prob[word][types[j]])
                    else:
                        V[t][j] = max(prob)+np.log(count)

        last_state = np.argmax(V[len(sentence) -1,:])

        path = []

        # Determining the path with most probability by back tracking
        path.append(types[last_state])
        index = 1
        for i in range(len(sentence) - 2,-1,-1):
            
            last_state = prev[i, int(last_state)]

            path.append(types[int(last_state)])
            
            index += 1
    
        path = np.flip(path, axis=0)

        return path


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")

