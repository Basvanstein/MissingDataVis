import getDataSet as ds

'''
{Post-oper.}      & 1   & ``NA''  & ``NA''  &``NA''          &  0.0333333333333 & {Monotone}\\
{Adult}           & 2   & ``NA''  & ``NA''  &``NA''          &  0.0741165390443 & {Two monotone patterns} \\
{Digits Mono} & 1   & ``binomial''  & ``binomial'' & ``random'' &  0.2 & {Monotone} \\
{Concrete Mono} & 1   & ``random''  & ``random'' & ``0.3''     &  0.4 & {Monotone} \\
{Page Mono}  & 2   & ``random''  & ``power'' & ``binomial''     &  0.6 & {Two monotone patterns} \\
\midrule
{Census}          & 7   & ``NA''  & ``NA''  &``NA''          &  0.5268          & {Mostly monotone}\\
{Hepatitis}       & 7   & ``NA''  & ``NA''  &``NA''          &  0.483870967742  & {70\% mono., 30\% rand.}\\
{Wiki}            & 116 & ``NA''  & ``NA''  &``NA''          &  0.807228915663  & {50\% mono., 50\% rand.}\\
{Digits Semi}     & 8   & ``binomial'' & ``random'' & ``binomial''  &  0.4 & {Semi random} \\
{Concrete Semi}   & 8   & ``power''   & ``power'' & ``random''         &  0.6 & {Semi random} \\
{Page Semi}       & 8   & ``random''  & ``binomial'' & ``0.5''    &  0.2 & {Semi random} \\
\midrule
{Marketing}       & 39  & ``NA''  & ``NA''  &``NA''          &  0.235405315245  & {Random}\\
{Horse-colic}     & 82  & ``NA''  & ``NA''  &``NA''          &  0.98097826087   & {Random}\\
{Digits Random}   & 100   & ``power'' & ``random'' & ``0.7''  &  0.4 & {Random} \\
{Concrete Random} & 100   & ``random''   & ``binomial'' & ``binomial''         &  0.6 & {Random} \\
{Page Random}     & 100   & ``binomial''  & ``power'' & ``random''    &  0.2 & {Random} \\
'''

Datasets = ["post-operative","adult","digits","concrete","page","census","hepatitis","wiki","digits","concrete","page","marketing","horse-colic","digits","concrete","page"]
Missing = [True,True,False,False,False,True,True,True,False,False,False,True,True,False,False,False]
K = [1,2,1,1,2,7,7,116,8,8,8,39,82,100,100,100]
MD = ["","","binomial","random","random","","","","binomial","power","random","","","power","random","binomial"]
PD = ["","","binomial","random","power","","","","random","power","binomial","","","random","binomial","power"]
RD = ["","","random","0.3","binomial","","","","binomial","random","0.5","","","0.7","binomial","random"]
PER = [0,0,20.0,40.0,60.0,0,0,0,40.0,60.0,20.0,0,0,40.0,60.0,20.0]
names = ["Post-oper.","Adult","Digits Mono","Concrete Mono","Page Mono","Census","Hepatitis","Wiki","Digits Semi","Concrete Semi","Page Semi","Marketing","Horse-colic","Digits Random","Concrete Random","Page Random"]

for i in range(len(names)):
	#print names[i]
	d,regression,Dim,amount,missing = ds.GetDataSet(dataname=Datasets[i])
	if (Missing[i]==False):
		RDI = RD[i]
		if (RDI=="0.5"):
			RDI = 0.5
		elif (RDI=="0.3"):
			RDI = 0.3
		elif (RDI=="0.7"):
			RDI = 0.7
		d.data = ds.generateMissingValuesNew(d.data,k=K[i],MixtureDistribution=MD[i],PatternDistribution=PD[i], RecordDistribution=RDI, Perc=PER[i] )

	#print d.data.shape
	mixtures,records, numbers = ds.analyzeDataset(d.data)
	maxlen = min(15,len(records))
	#for i in range(len(mixtures)):
		#print "Mixture", mixtures[i], "len records:", numbers[i]
	numbers = np.array(numbers)
	percent_numbers = numbers.astype(float) / np.sum(numbers)
	totoal_perc =  np.sum(numbers) / float( len(d.data) )

	print names[i], "&",len(mixtures),"&", percent_numbers, "&", totoal_perc, "\\\\"

	ds.plotPatterns(records,'benchmark img/mixtures'+`i`+'.png',True,maxlen)
	ds.plotPatternsInRecords(mixtures,records,'benchmark img/mapping'+`i`+'.png',True,maxlen)
