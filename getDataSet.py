import numpy as np
from deap import benchmarks
from sklearn import datasets
from sklearn.datasets import fetch_mldata, make_regression, make_classification
from sklearn import cross_validation
from sklearn.preprocessing import Imputer
from collections import defaultdict
import math


def diffpow(x):
	N = x.shape[0]
	x = np.abs(x)
	p = (10*np.arange(0, N) / (N-1.0) + 2).reshape(-1, 1)
	f = np.sum(np.power(x, p), axis=0)
	return f

def ackley_arg0(sol):
	X,Y = sol[0], sol[1]
	Z = np.zeros(X.shape)
	
	for i in xrange(X.shape[0]):
		Z[i] = benchmarks.ackley((X[i],Y[i]))[0]
	return Z

def himmelblau_arg0(sol):
	X,Y = sol[0], sol[1]
	Z = np.zeros(X.shape)
	
	for i in xrange(X.shape[0]):
		Z[i] = benchmarks.himmelblau((X[i],Y[i]))[0]
	return Z


def rastrigin_arg10(sol):
	#D1,D2,D3,D4,D5,D6,D7,D8,D9,D10 = sol[0], sol[1], sol[2], sol[3], sol[4], sol[5], sol[6], sol[7], sol[8], sol[9]
	Z = np.zeros(sol.shape[0])
	#print sol.shape[0]
	for i in xrange(sol.shape[0]):
		#print sol[i]
		Z[i] = benchmarks.rastrigin(sol[i])[0]
		#print Z[i]
	return Z

def rastrigin_arg0(sol):
	X,Y = sol[0], sol[1]
	Z = np.zeros(X.shape)
	
	for i in xrange(X.shape[0]):
		Z[i] = benchmarks.rastrigin((X[i],Y[i]))[0]
	return Z

def schwefel_arg0(sol):
	X,Y = sol[0], sol[1]
	Z = np.zeros(X.shape)
	
	for i in xrange(X.shape[0]):
		Z[i] = benchmarks.schwefel((X[i],Y[i]))[0]
	return Z

def schaffer_arg0(sol):
	X,Y = sol[0], sol[1]
	Z = np.zeros(X.shape)
	
	for i in xrange(X.shape[0]):
		Z[i] = benchmarks.schaffer((X[i],Y[i]))[0]
	return Z

A = [[0.5, 0.5], [0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]
C = [0.002, 0.005, 0.005, 0.005, 0.005]

def shekel_arg0(sol):
	X,Y = sol[0], sol[1]
	Z = np.zeros(X.shape)
	
	for i in xrange(X.shape[0]):
		Z[i] = benchmarks.shekel((X[i],Y[i]),A,C)[0]
	return Z

def h1_arg0(sol):
	X,Y = sol[0], sol[1]
	Z = np.zeros(X.shape)
	
	for i in xrange(X.shape[0]):
		Z[i] = benchmarks.h1((X[i],Y[i]))[0]
	return Z

def rosenbrock_arg0(sol):
	X,Y = sol[0], sol[1]
	Z = np.zeros(X.shape)
	
	for i in xrange(X.shape[0]):
		Z[i] = benchmarks.rosenbrock((X[i],Y[i]))[0]
	return Z




def generateMissingValues(X_train, test_x, missing_perc=20,mode="normal"):

	end_column_with_missing_data = len(X_train[0])
	random_columns = np.arange(end_column_with_missing_data)
	np.random.shuffle(random_columns)

	print mode, random_columns

	missing=0
	j_size = len(X_train)
	max_missing = j_size * (len(X_train[0]) /2) * missing_perc / 100 #be sure to keep enough complete training points

	while (missing < max_missing):
		i = np.random.randint(0,j_size/2)
		if mode == "random":
			j = np.random.randint(0,end_column_with_missing_data)
		if mode == "normal" or mode == "mono":
			n, p = end_column_with_missing_data, .5 # number of trials, probability of each trial
			j = np.random.binomial(n, p)
		if j < 0:
			j = 0
		if j >= end_column_with_missing_data:
			j = end_column_with_missing_data-1

		if mode == "mono":

			for jsmall in range(j):
				temp = random_columns[jsmall]
				if (np.isnan(X_train[i][temp]) == True):
					break
				else:
					missing += 1
					#print temp
					X_train[i][temp] = np.nan

		else:
			j = random_columns[j]
			if (np.isnan(X_train[i][j]) == False):
				missing += 1
				X_train[i][j] = np.nan

	missing=0
	j_size = len(test_x)
	max_missing = j_size * len(test_x[0]) * missing_perc / 100


	#now add missing values to the test set

	

	while (missing < max_missing):
		i = np.random.randint(0,j_size)
		if mode == "random":
			j = np.random.randint(0,end_column_with_missing_data) 
		if mode == "normal" or mode == "mono":
			n, p = end_column_with_missing_data, .5 # number of trials, probability of each trial
			j = np.random.binomial(n, p)
		if j < 0:
			j = 0
		if j >= end_column_with_missing_data:
			j = end_column_with_missing_data-1
		
		if mode == "mono":
			for jsmall in range(j):
				temp = random_columns[jsmall]
				if (np.isnan(test_x[i][temp]) == True):
					break
				else:
					missing += 1
					test_x[i][temp] = np.nan

		else:
			j = random_columns[j]
			if (np.isnan(test_x[i][j]) == False):
				missing += 1
				test_x[i][j] = np.nan

	return X_train, test_x


def createTraingSet(datasetname,seed, missing, test_size=0.5):
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(datasetname.data, datasetname.target, test_size=test_size, random_state=seed)
	#if len(X_train) > 5000:
	#	X_train = X_train[:5000,:]
	#	y_train = y_train[:5000]
	#imp = Imputer(missing_values='NaN', strategy='median', axis=1)
	#imp.fit(X_train)
	#X_train_imp = imp.transform(X_train)
	#print X_train.shape
	#a = np.vstack([X_train.T,y_train.T]).T
	#Everything = a[~np.isnan(a).any(axis=1)]
	#X_train_imp = Everything[:,:-1]
	#y_train = Everything[:,-1]
	#print "training set:", X_train_imp.shape
	#exit()
	return X_train, X_test, y_train, y_test


def convertLMH(x):
	if x == "low":
		return 0
	elif x == "mid":
		return 1
	elif x == "high":
		return 2
def convertGE(x):
	if x == "excellent":
		return 1
	elif x == "good":
		return 0

def SURFSTBL(x):
	if x == "stable":
		return 0
	elif x == "unstable":
		return 1
	elif x == "mod-stable":
		return 2
def ASI(x):
	if x == "A":
		return 0
	elif x == "S":
		return 1
	elif x == "I":
		return 2
def convertBAND(x):
	if x == "band":
		return 1
	else:
		return 0



def convertAdult_1(x):
	n1 = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]
	return n1.index(x)
def convertAdult_3(x):
	n1 = ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]
	return n1.index(x)
def convertAdult_5(x):
	n1 = ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
	return n1.index(x)
def convertAdult_6(x):
	n1 = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]
	return n1.index(x)

def convertAdult_7(x):
	n1 = ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
	return n1.index(x)
def convertAdult_8(x):
	n1 = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
	return n1.index(x)
def convertAdult_9(x):
	n1 = ["Female", "Male"]
	return n1.index(x)
def convertAdult_13(x):
	n1 = ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]
	return n1.index(x)
def convertAdult_14(x):
	n1 = [">50K", "<=50K"]
	return n1.index(x)

def Census_1(x):
	n1 = ["Self-employed-not_incorporated","Not_in_universe","Private","Local_government","Federal_government","Self-employed-incorporated","State_government","Never_worked","Without_pay"]
	return n1.index(x)
def Census_4(x):
	n1 = ["Some_college_but_no_degree","10th_grade","Children","Bachelors_degree(BA_AB_BS)","High_school_graduate","Masters_degree(MA_MS_MEng_MEd_MSW_MBA)","Less_than_1st_grade","Associates_degree-academic_program","7th_and_8th_grade","12th_grade_no_diploma","Associates_degree-occup_/vocational","Prof_school_degree_(MD_DDS_DVM_LLB_JD)","5th_or_6th_grade","11th_grade","Doctorate_degree(PhD_EdD)","9th_grade","1st_2nd_3rd_or_4th_grade"]
	return n1.index(x)
def Census_6(x):
	n1 = ["Not_in_universe","High_school","College_or_university"]
	return n1.index(x)
def Census_7(x):
	n1 = ["Divorced","Never_married","Married-civilian_spouse_present","Widowed","Separated","Married-spouse_absent","Married-A_F_spouse_present"]
	return n1.index(x)
def Census_8(x):
	n1 = ["Construction","Not_in_universe_or_children","Entertainment","Finance_insurance_and_real_estate","Education","Business_and_repair_services","Manufacturing-nondurable_goods","Personal_services_except_private_HH","Manufacturing-durable_goods","Other_professional_services","Mining","Transportation","Wholesale_trade","Public_administration","Retail_trade","Social_services","Private_household_services","Utilities_and_sanitary_services","Communications","Hospital_services","Medical_except_hospital","Agriculture","Forestry_and_fisheries","Armed_Forces"]
	return n1.index(x)
def Census_9(x):
	n1 = ["Precision_production_craft_&_repair","Not_in_universe","Professional_specialty","Executive_admin_and_managerial","Handlers_equip_cleaners_etc","Adm_support_including_clerical","Machine_operators_assmblrs_&_inspctrs","Other_service","Sales","Private_household_services","Technicians_and_related_support","Transportation_and_material_moving","Farming_forestry_and_fishing","Protective_services","Armed_Forces"]
	return n1.index(x)
def Census_10(x):
	n1 = ["White","Asian_or_Pacific_Islander","Amer_Indian_Aleut_or_Eskimo","Black","Other"]
	return n1.index(x)
def Census_11(x):
	n1 = ["All_other","Do_not_know","Central_or_South_American","Mexican_(Mexicano)","Mexican-American","Other_Spanish","Puerto_Rican","Cuban","Chicano","NA"]
	return n1.index(x)
def Census_12(x):
	n1 = ["Male","Female"]
	return n1.index(x)
def Census_13(x):
	n1 = ["Not_in_universe","No","Yes"]
	return n1.index(x)
def Census_14(x):
	n1 = ["Not_in_universe","Job_loser_-_on_layoff","Other_job_loser","New_entrant","Re-entrant","Job_leaver"]
	return n1.index(x)
def Census_15(x):
	n1 = ["Children_or_Armed_Forces","Not_in_labor_force","Full-time_schedules","Unemployed_full-time","Unemployed_part-_time","PT_for_non-econ_reasons_usually_FT","PT_for_econ_reasons_usually_PT","PT_for_econ_reasons_usually_FT"]
	return n1.index(x)
def Census_19(x):
	n1 = ["Head_of_household","Nonfiler","Joint_both_under_65","Single","Joint_both_65+","Joint_one_under_65_&_one_65+"]
	return n1.index(x)
def Census_20(x):
	n1 = ["South","Not_in_universe","Northeast","Midwest","West","Abroad"]
	return n1.index(x)
def Census_21(x):
	n1 = ["Arkansas","Not_in_universe","Utah","Michigan","Minnesota","Alaska","Kansas","Indiana","Massachusetts","New_Mexico","Nevada","Tennessee","Colorado","Abroad","Kentucky","California","Arizona","North_Carolina","Connecticut","Florida","Vermont","Maryland","Oklahoma","Oregon","Ohio","South_Carolina","Texas","Montana","Wyoming","Georgia","Pennsylvania","Iowa","New_Hampshire","Missouri","Alabama","North_Dakota","New_Jersey","Louisiana","West_Virginia","Delaware","Illinois","Maine","Wisconsin","New_York","Idaho","District_of_Columbia","South_Dakota","Nebraska","Virginia","Mississippi"]
	return n1.index(x)
def Census_22(x):
	n1 = ["Householder","Child_18+_never_marr_Not_in_a_subfamily","Child_<18_never_marr_not_in_subfamily","Spouse_of_householder","Secondary_individual","Other_Rel_18+_never_marr_not_in_subfamily","Nonfamily_householder","Grandchild_<18_never_marr_not_in_subfamily","Grandchild_<18_never_marr_child_of_subfamily_RP","Child_18+_ever_marr_Not_in_a_subfamily","Child_18+_never_marr_RP_of_subfamily","Child_18+_spouse_of_subfamily_RP","Other_Rel_<18_never_marr_child_of_subfamily_RP","Child_under_18_of_RP_of_unrel_subfamily","Grandchild_18+_never_marr_not_in_subfamily","Child_18+_ever_marr_RP_of_subfamily","Other_Rel_18+_ever_marr_RP_of_subfamily","Other_Rel_18+_ever_marr_not_in_subfamily","RP_of_unrelated_subfamily","Other_Rel_18+_spouse_of_subfamily_RP","Other_Rel_<18_never_marr_not_in_subfamily","Other_Rel_<18_spouse_of_subfamily_RP","In_group_quarters","Grandchild_18+_spouse_of_subfamily_RP","Other_Rel_18+_never_marr_RP_of_subfamily","Child_<18_never_marr_RP_of_subfamily","Child_<18_ever_marr_not_in_subfamily","Other_Rel_<18_ever_marr_RP_of_subfamily","Grandchild_18+_ever_marr_not_in_subfamily","Child_<18_spouse_of_subfamily_RP","Spouse_of_RP_of_unrelated_subfamily","Other_Rel_<18_never_married_RP_of_subfamily","Grandchild_18+_never_marr_RP_of_subfamily","Grandchild_18+_ever_marr_RP_of_subfamily","Child_<18_ever_marr_RP_of_subfamily","Other_Rel_<18_ever_marr_not_in_subfamily","Grandchild_<18_never_marr_RP_of_subfamily","Grandchild_<18_ever_marr_not_in_subfamily"]
	return n1.index(x)
def Census_23(x):
	n1 = ["Householder","Child_18_or_older","Child_under_18_never_married","Spouse_of_householder","Nonrelative_of_householder","Other_relative_of_householder","Group_Quarters-_Secondary_individual","Child_under_18_ever_married"]
	return n1.index(x)
def Census_25(x):
	n1 = ["MSA_to_MSA","Nonmover","NonMSA_to_nonMSA","Not_in_universe","Not_identifiable","Abroad_to_MSA","MSA_to_nonMSA","Abroad_to_nonMSA","NonMSA_to_MSA"]
	return n1.index(x)
def Census_26(x):
	n1 = ["Same_county","Nonmover","Different_region","Different_county_same_state","Not_in_universe","Different_division_same_region","Abroad","Different_state_same_division"]
	return n1.index(x)
def Census_27(x):
	n1 = ["Same_county","Nonmover","Different_state_in_South","Different_county_same_state","Not_in_universe","Different_state_in_Northeast","Abroad","Different_state_in_Midwest","Different_state_in_West"]
	return n1.index(x)
def Census_28(x):
	n1 = ["No","Not_in_universe_under_1_year_old","Yes"]
	return n1.index(x)
def Census_29(x):
	n1 = ["Yes","Not_in_universe","No"]
	return n1.index(x)
def Census_31(x):
	n1 = ["Not_in_universe","Both_parents_present","Mother_only_present","Neither_parent_present","Father_only_present"]
	return n1.index(x)
def Census_32(x):
	n1 = ["United-States","Vietnam","Philippines","Columbia","Germany","Mexico","Japan","Peru","Dominican-Republic","South_Korea","Cuba","El-Salvador","Canada","Scotland","Outlying-U_S_(Guam_USVI_etc)","Italy","Guatemala","Ecuador","Puerto-Rico","Cambodia","China","Poland","Nicaragua","Taiwan","England","Ireland","Hungary","Yugoslavia","Trinadad&Tobago","Jamaica","Honduras","Portugal","Iran","France","India","Hong_Kong","Haiti","Greece","Holand-Netherlands","Thailand","Laos","Panama"]
	return n1.index(x)
def Census_33(x):
	n1 = ["United-States","Vietnam","Columbia","Mexico","El-Salvador","Peru","Puerto-Rico","Cuba","Philippines","Dominican-Republic","Germany","England","Guatemala","Scotland","Portugal","Italy","Ecuador","Yugoslavia","China","Poland","Hungary","Nicaragua","Taiwan","Ireland","Canada","South_Korea","Trinadad&Tobago","Jamaica","Honduras","Iran","France","Cambodia","India","Hong_Kong","Haiti","Japan","Greece","Holand-Netherlands","Thailand","Panama","Laos","Outlying-U_S_(Guam_USVI_etc)"]
	return n1.index(x)
def Census_34(x):
	n1 = ["United-States","Vietnam","Columbia","Mexico","Peru","Cuba","Philippines","Dominican-Republic","El-Salvador","Canada","Scotland","Portugal","Guatemala","Ecuador","Germany","Outlying-U_S_(Guam_USVI_etc)","Puerto-Rico","Italy","China","Poland","Nicaragua","Taiwan","England","Ireland","South_Korea","Trinadad&Tobago","Jamaica","Honduras","Iran","Hungary","France","Cambodia","India","Hong_Kong","Japan","Haiti","Holand-Netherlands","Greece","Thailand","Panama","Yugoslavia","Laos"]
	return n1.index(x)
def Census_35(x):
	n1 = ["Native-_Born_in_the_United_States","Foreign_born-_Not_a_citizen_of_U_S","Foreign_born-_U_S_citizen_by_naturalization","Native-_Born_abroad_of_American_Parent(s)","Native-_Born_in_Puerto_Rico_or_U_S_Outlying"]
	return n1.index(x)
def Census_37(x):
	n1 = ["Not_in_universe","No","Yes"]
	return n1.index(x)
def Census_CLASS(x):
	n1 = ["-_50000.","50000+."]
	return n1.index(x)


def auto_1(x):
	n1 = ["alfa-romero","audi","bmw","chevrolet","dodge","honda","isuzu","jaguar","mazda","mercedes-benz","mercury","mitsubishi","nissan","peugot","plymouth","porsche","renault","saab","subaru","toyota","volkswagen","volvo"]
	return n1.index(x)
def auto_2(x):
	n1 = ["diesel","gas"]
	return n1.index(x)
def auto_3(x):
	n1 = ["std","turbo"]
	return n1.index(x)
def auto_4(x):
	n1 = ["four","two"]
	return n1.index(x)
def auto_5(x):
	n1 = ["hardtop","wagon","sedan","hatchback","convertible"]
	return n1.index(x)
def auto_6(x):
	n1 = ["4wd","fwd","rwd"]
	return n1.index(x)
def auto_7(x):
	n1 = ["front","rear"]
	return n1.index(x)
def auto_13(x):
	n1 = ["dohc","dohcv","l","ohc","ohcf","ohcv","rotor"]
	return n1.index(x)
def auto_14(x):
	n1 = ["eight","five","four","six","three","twelve","two"]
	return n1.index(x)
def auto_16(x):
	n1 = ["1bbl","2bbl","4bbl","idi","mfi","mpfi","spdi","spfi"]
	return n1.index(x)

#import time
#import datetime
def convertDate(x):
	return 0 #time.mktime(datetime.datetime.strptime(x, "%d/%m/%Y").timetuple())

def GetDataSet(dataname="rast",amount=1000, Dim=2):
	missing = False
	d = None

	datasets_with_missing_data = ["wiki","automobile","post-operative","wisconsin","marketing","dermatology","mammographic","horse-colic","hepatitis","cleveland","bands", "adult","census"]#,"automobile","bands","breast","census","cleveland","crx","dermatology","hepatitis","horse-colic","housevotes","mammographic","marketing","mushroom",]

	#print dataname
	if dataname == "allhouses":
		d = fetch_mldata('uci-20070111 house_16H')
	#	print d.data
		d.target = d.data[:,(len(d.data[0])-1)]
	#	print d.target
		d.data = d.data[:,:(len(d.data[0])-1)]

		regression = True

	elif dataname in datasets_with_missing_data:
		d = lambda:0 
		regression = False
		converters = {}
		if dataname == "bands":
			converters = {19:convertBAND}
		if dataname == "post-operative":
			converters = {0:convertLMH,1:convertLMH,2:convertGE,3:convertLMH,4:SURFSTBL,5:SURFSTBL,6:SURFSTBL,8:ASI}
		if dataname == "adult":
			converters = {1:convertAdult_1, 3:convertAdult_3, 5:convertAdult_5,6:convertAdult_6,7:convertAdult_7,8:convertAdult_8,9:convertAdult_9,13:convertAdult_13,14:convertAdult_14}
		if dataname == "census":
			converters = {1:Census_1, 4:Census_4, 6:Census_6, 7:Census_7, 8:Census_8, 9:Census_9, 10:Census_10, 11:Census_11,12:Census_12,13:Census_13,14:Census_14,15:Census_15, 19:Census_19,20:Census_20,21:Census_21,22:Census_22,23:Census_23,25:Census_25,26:Census_26,27:Census_27,28:Census_28,29:Census_29,31:Census_31, 32:Census_32,33:Census_33,34:Census_34,35:Census_35,37:Census_37,41:Census_CLASS}

		if dataname == "automobile":
			converters = {1:auto_1,2:auto_2,3:auto_3,4:auto_4,5:auto_5,6:auto_6,7:auto_7,13:auto_13,14:auto_14,16:auto_16}
			#regression = True
		
		if dataname == "wiki":
			regression = True
			d.data = np.genfromtxt("datasets/missing/wiki4HE.csv", delimiter=";", comments="@", missing_values=["?","<null>"],filling_values=np.nan, converters=converters)
			d.target = d.data[:,0]
			d.data = d.data[:,1:]
		else:

			d.data = np.genfromtxt("datasets/missing/"+dataname+".dat", delimiter=",", comments="@", missing_values=["?","<null>"],filling_values=np.nan, converters=converters)
			if dataname == "census":
				d.data = d.data[:20000,:]
			d.target = d.data[:,(len(d.data[0])-1)]
			d.data = d.data[:,:(len(d.data[0])-1)]

		
		missing = True

	elif dataname == "ozone":
		d = lambda:0 
		
		converters = {0:convertDate}
		
		d.data = np.genfromtxt("datasets/missing/ozone/eighthr.data", delimiter=",", comments="@", missing_values=["?","<null>"],filling_values=np.nan, converters=converters)
		d.target = d.data[:,(len(d.data[0])-1)]
		d.data = d.data[:,1:(len(d.data[0])-1)]
		#print d.data
		#print d.data.shape
		#print d.target
		#exit()
		regression = False
		missing = True



	elif dataname == "eye":
		d = lambda:0 
		# https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State
		d.data = np.loadtxt("datasets/Eye/EEG Eye State.csv", delimiter=",")
		#print(d.data.shape)

		d.target = d.data[:,(len(d.data[0])-1)]
		d.data = d.data[:,:(len(d.data[0])-1)]
		regression = False

	elif dataname == "page":
		d = lambda:0 
		# https://archive.ics.uci.edu/ml/datasets/Page+Blocks+Classification
		d.data = np.loadtxt("datasets/page-blocks/page-blocks.data", delimiter=",")
		#print(d.data.shape)

		d.target = d.data[:,(len(d.data[0])-1)]
		d.data = d.data[:,:(len(d.data[0])-1)]
		regression = False
		
		#print d.data.shape
		#exit()
	elif dataname == "skill": #fails to load
		d = lambda:0 
		# https://archive.ics.uci.edu/ml/datasets/SkillCraft1+Master+Table+Dataset
		d.data = np.loadtxt("datasets/SkillCraft/SkillCraft1_Dataset.csv", delimiter=",")
		#print(d.data.shape)

		d.target = d.data[:,(len(d.data[0])-1)]
		d.data = d.data[:,:(len(d.data[0])-1)]
		regression = True
		
		#print d.data.shape
		exit()

	elif dataname == "concrete":
		d = lambda:0 
		# https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
		d.data = np.loadtxt("datasets/Concrete/Concrete_Data.csv", delimiter=",")
		#print(d.data.shape)

		d.target = d.data[:,(len(d.data[0])-1)]
		d.data = d.data[:,:(len(d.data[0])-1)]
		regression = True
		
		#print d.data.shape
		#exit()

		

	elif dataname == "digits":
		d = datasets.load_digits()
		regression = False
	elif dataname == "iris":#too small
		d = datasets.load_iris()
		regression = False
		
	elif dataname == "cover":
		d = datasets.fetch_covtype()
		regression = False


	elif dataname == "makereg":

		X, y = make_regression(n_samples=amount, n_features=Dim, n_informative=Dim-2, n_targets=1, bias=0.0)
		d = lambda:0 
		d.data = X
		d.target = y
		regression = True
	elif dataname == "makeclass":

		X, y = make_classification(n_samples=amount, n_features=Dim, n_informative=Dim-2)
		d = lambda:0 
		d.data = X
		d.target = y
		regression = False

	elif dataname == "rast":
		#rastrigin_arg0
		d = lambda:0 
		X2 = np.random.uniform(low=-4.9,high=4.9,size=(Dim,amount))
		d.data = X2.T
		#X_test2 = np.random.uniform(low=-lowv1,high=highv1,size=(10,n))
		d.target = rastrigin_arg0(d.data.T)
		regression = True
	elif dataname == "ackley":
		#rastrigin_arg0
		d = lambda:0 
		X2 = np.random.uniform(low=-29.9,high=29.9,size=(Dim,amount))
		d.data = X2.T
		#X_test2 = np.random.uniform(low=-lowv1,high=highv1,size=(10,n))
		d.target = ackley_arg0(d.data.T)
		regression = True
	elif dataname == "himmelblau":
		#rastrigin_arg0
		d = lambda:0 
		X2 = np.random.uniform(low=-5.9,high=5.9,size=(Dim,amount))
		d.data = X2.T
		#X_test2 = np.random.uniform(low=-lowv1,high=highv1,size=(10,n))
		d.target = himmelblau_arg0(d.data.T)
		regression = True
	elif dataname == "schwefel":
		#rastrigin_arg0
		d = lambda:0 
		X2 = np.random.uniform(low=-499.9,high=499.9,size=(Dim,amount))
		d.data = X2.T
		#X_test2 = np.random.uniform(low=-lowv1,high=highv1,size=(10,n))
		d.target = schwefel_arg0(d.data.T)
		regression = True
	elif dataname == "h1":
		#rastrigin_arg0
		d = lambda:0 
		X2 = np.random.uniform(low=-24.9,high=24.9,size=(Dim,amount))
		d.data = X2.T
		#X_test2 = np.random.uniform(low=-lowv1,high=highv1,size=(10,n))
		d.target = h1_arg0(d.data.T)
		regression = True
	elif dataname == "rosenbrock":
		#rastrigin_arg0
		d = lambda:0 
		X2 = np.random.uniform(low=-1.99,high=1.99,size=(Dim,amount))
		d.data = X2.T
		#X_test2 = np.random.uniform(low=-lowv1,high=highv1,size=(10,n))
		d.target = rosenbrock_arg0(d.data.T)
		regression = True
	elif dataname == "schaffer":
		#rastrigin_arg0
		d = lambda:0 
		X2 = np.random.uniform(low=-24.9,high=24.9,size=(Dim,amount))
		d.data = X2.T
		#X_test2 = np.random.uniform(low=-lowv1,high=highv1,size=(10,n))
		d.target = schaffer_arg0(d.data.T)
		regression = True
	elif dataname == "diffpow":
		#rastrigin_arg0
		d = lambda:0 
		X2 = np.random.uniform(low=-24.9,high=24.9,size=(Dim,amount))
		d.data = X2.T
		#X_test2 = np.random.uniform(low=-lowv1,high=highv1,size=(10,n))
		d.target = diffpow(d.data.T)
		regression = True
	elif dataname == "europe":
		#rastrigin_arg0
		from PIL import Image
		im = Image.open('datasets/elevationdata/europe.tif')
		#im.show()
		imarray = np.array(im)
		d = lambda:0 
		d.data = []
		d.target = []
		

		print imarray.shape
		for i in range(len(imarray)):
			for j in range(len(imarray[i])):
				d.data.append([i,j])
				d.target.append(imarray[i][j])
		d.data = np.array(d.data).astype('float')
		d.target = np.array(d.target).astype('float')

		if len(d.data) > amount:
			everything = np.vstack([d.data.T,d.target.T]).T
			np.random.shuffle(everything)
			d.data = everything[:amount,:-1]
			d.target = everything[:amount,-1]
			d.test_data = everything[0::100,:-1]
			d.test_target = everything[0::100,-1]

		regression = True
	else:
		print "No such dataset";
		exit()
	Dim = len(d.data[0])
	amount = len(d.data)
	return d,regression,Dim,amount,missing







##########################################################
# MISSING DATA BENCHMARK PAPER
#
#########################################################

def analyzeDatasetBruteForce(Data, missing_indicator=np.nan):
	#bruteforce units
	all_mixtures = []
	all_nice_mixtures = []
	all_mixture_records = []
	all_num_mixture_records = []

	mixtures = []
	nice_mixtures = []
	records_done = []
	

	nan_set_indexes = dict()
	nan_sets = defaultdict(int)
	nan_support = defaultdict(int)
	modelarray = dict()
	modelindexarray = dict()
	totalmissing = 0

	attributes = range(len(Data[0]));
	for i in range(len(Data)):
		nan_indices = ""
		nan_found = 0
		#a_indices = []
		for j in attributes: 
			if (math.isnan(Data[i][j]) ):
				#a_indices.append(j)
				nan_indices = nan_indices + " " + `j`
		if (nan_indices!=""):
			totalmissing+=1
			if nan_indices not in nan_set_indexes:
				nan_set_indexes[nan_indices] = []
			nan_sets[nan_indices] += 1
			nan_set_indexes[nan_indices].append(i)

	#print "unsorted:", nan_sets
	sorted_nan = sorted(nan_sets, key=nan_sets.get, reverse=True)
	#print "Sorted", sorted_nan
	

	all_mixtures.append([])

	mixture_records = []
	all_mixture_records.append(mixture_records)
	all_nice_mixtures.append(nice_mixtures)
	num_mixture_records = []
	all_num_mixture_records.append(num_mixture_records)
	D = Data

	for set in sorted_nan:
		#set is the starting set of our monotone missing data pattern
		#it either belongs to an already existing pattern, or it is a new pattern.
		nan_inds = map(int, set.split())


		set_part_of_mixture = False
		for mi in range(len(all_mixtures)):
			#all_mixtures = []
			#all_nice_mixtures = []
			#all_mixture_records = []
			mixtures = all_mixtures[mi]
			mixture_records = all_mixture_records[mi]
			nice_mixtures = all_nice_mixtures[mi]
			num_mixture_records = all_num_mixture_records[mi]
			add_new_possibility = False

			#first copy mixture to temp vars
			temp_mixture_records = copy.deepcopy(mixture_records)
			temp_mixtures = copy.deepcopy(mixtures)
			temp_nice_mixtures = copy.deepcopy(nice_mixtures)
			temp_num_mixture_records = copy.deepcopy(num_mixture_records)

			for i in range(len(temp_mixtures)):
				mixture_sets = temp_mixtures[i]
				set_part_of_mixture = True
				for mset in mixture_sets:
					toCheckMixture = map(int, mset.split())
					MixtureSubsetOfSet = True
					for c in toCheckMixture:
						if (c not in nan_inds):
							MixtureSubsetOfSet = False
							break
					SetSubsetOfMixture = True
					for c in nan_inds:
						if (c not in toCheckMixture):
							SetSubsetOfMixture = False
							break
					if (SetSubsetOfMixture==False and MixtureSubsetOfSet==False ):
						set_part_of_mixture = False
						break


				if (set_part_of_mixture == True and add_new_possibility == False):
					mixture_records[i].append(nan_set_indexes[set])
					mixtures[i].append(set)
					nice_mixtures[i].append(nan_inds)
					num_mixture_records[i] += len(nan_set_indexes[set])
					add_new_possibility = True
				elif (set_part_of_mixture == True and add_new_possibility == True and len(all_mixtures)<100000):

					#add a possible mixture model
					new_mixture_records = copy.deepcopy(temp_mixture_records)
					new_mixtures = copy.deepcopy(temp_mixtures)
					new_nice_mixtures = copy.deepcopy(temp_nice_mixtures)
					new_num_mixture_records  = copy.deepcopy(temp_num_mixture_records)

					new_mixture_records[i].append(nan_set_indexes[set])
					new_mixtures[i].append(set)
					new_nice_mixtures[i].append(nan_inds)
					new_num_mixture_records[i] += len(nan_set_indexes[set])


					all_mixtures.append(new_mixtures)
					all_mixture_records.append(new_mixture_records)
					all_nice_mixtures.append(new_nice_mixtures)
					all_num_mixture_records.append(new_num_mixture_records)
					#print len(all_mixtures)

			if (set_part_of_mixture == False and add_new_possibility == False):
				#create new mixture
				mixtures.append([set])
				nice_mixtures.append([nan_inds])
				mixture_records.append([nan_set_indexes[set]])
				num_mixture_records.append(len(nan_set_indexes[set]))
		
	#for i in range(len(mixtures)):
		#print "Mixture", nice_mixtures[i], "len records:", num_mixture_records[i]
		
	#returns: mixtures (every record is a mixture, consisting of sets (arrays) indicating the combinations of missing attributes in the mixture.)
	# mixture_records: every row is an array of arrays with the indeces of the dataset for each combination of missing attributes of the mixtures[i]
	# num_mixture_records: the amount of records per mixture
	
	return all_nice_mixtures, all_mixture_records, all_num_mixture_records

# analyze a dataset to find out how many monotone patterns of missing data might be in the dataset
def analyzeDataset(Data, missing_indicator=np.nan):
	mixtures = []
	nice_mixtures = []
	records_done = []
	

	nan_set_indexes = dict()
	nan_sets = defaultdict(int)
	nan_support = defaultdict(int)
	modelarray = dict()
	modelindexarray = dict()
	totalmissing = 0

	attributes = range(len(Data[0]));
	for i in range(len(Data)):
		nan_indices = ""
		nan_found = 0
		#a_indices = []
		for j in attributes: 
			if (math.isnan(Data[i][j]) ):
				#a_indices.append(j)
				nan_indices = nan_indices + " " + `j`
		if (nan_indices!=""):
			totalmissing+=1
			if nan_indices not in nan_set_indexes:
				nan_set_indexes[nan_indices] = []
			nan_sets[nan_indices] += 1
			nan_set_indexes[nan_indices].append(i)

	#print "unsorted:", nan_sets
	sorted_nan = sorted(nan_sets, key=nan_sets.get, reverse=True)
	#print "Sorted", sorted_nan
	

	mixture_records = []
	num_mixture_records = []
	D = Data

	for set in sorted_nan:
		#set is the starting set of our monotone missing data pattern
		#it either belongs to an already existing pattern, or it is a new pattern.
		nan_inds = map(int, set.split())


		set_part_of_mixture = False

		for i in range(len(mixtures)):
			mixture_sets = mixtures[i]
			set_part_of_mixture = True
			for mset in mixture_sets:
				toCheckMixture = map(int, mset.split())
				MixtureSubsetOfSet = True
				for c in toCheckMixture:
					if (c not in nan_inds):
						MixtureSubsetOfSet = False
						break
				SetSubsetOfMixture = True
				for c in nan_inds:
					if (c not in toCheckMixture):
						SetSubsetOfMixture = False
						break
				if (SetSubsetOfMixture==False and MixtureSubsetOfSet==False ):
					set_part_of_mixture = False
					break


			if (set_part_of_mixture == True):
				mixture_records[i].append(nan_set_indexes[set])
				mixtures[i].append(set)
				nice_mixtures[i].append(nan_inds)
				num_mixture_records[i] += len(nan_set_indexes[set])
				break
		if (set_part_of_mixture == False):
			#create new mixture
			mixtures.append([set])
			nice_mixtures.append([nan_inds])
			mixture_records.append([nan_set_indexes[set]])
			num_mixture_records.append(len(nan_set_indexes[set]))
	
	#for i in range(len(mixtures)):
		#print "Mixture", nice_mixtures[i], "len records:", num_mixture_records[i]
		
	#returns: mixtures (every record is a mixture, consisting of sets (arrays) indicating the combinations of missing attributes in the mixture.)
	# mixture_records: every row is an array of arrays with the indeces of the dataset for each combination of missing attributes of the mixtures[i]
	# num_mixture_records: the amount of records per mixture
	return nice_mixtures, mixture_records, num_mixture_records

from matplotlib import cm, colors
import matplotlib.pyplot as pl
import copy
#
# Plot the various mixture patterns against the records
def plotPatternsInRecords(mixtures,records, filename="", save=False, maxlen=None):
	if (maxlen==None):
		maxlen = len(records)
	fig, ax = pl.subplots()
	colors=cm.Dark2(np.linspace(0,1,maxlen+1))
	index = 0;
	i = 0
	j = 0
	width = 1.0
	for m in records:
		mix = mixtures[i]
		longest = float(len(max(mix,key=len)))
		m_index = 0
		for m_i in m:
			le = len(mix[m_index])
			m_index += 1
			j += 1
			w = np.ones(len(m_i)) * float(le / longest)
			ax.barh(bottom=m_i, width=w, left=i, height=1.0, color=colors[i],edgecolor = "none")
		i+=1
		j += 1
		if (i > maxlen):
			break;
	ax.set_ylabel('record')
	ax.set_xlabel('')
	ax.set_xticks([])
	pl.gca().invert_yaxis()
	if (save==True):
		pl.savefig(filename, bbox_inches='tight')
	else:
		pl.show()
	pl.clf()


#
# Plot the various mixture patterns
def plotPatterns(records, filename="", save=False, maxlen=None):
	if (maxlen==None):
		maxlen = len(records)
	fig, ax = pl.subplots()
	colors=cm.Dark2(np.linspace(0,1,maxlen+1))
	index = 0;
	i = 0
	j = 0
	width = 1
	for m in records:
		for m_i in m:
			j += 1

			ax.bar(j, [len(m_i)], width, color=colors[i],edgecolor = "none")
		i+=1
		j += 1
		if (i > maxlen):
			break;
	ax.set_ylabel('number of records')
	ax.set_xlabel('Monotone Mixture Patterns')
	ax.set_xticks([])
	if (save==True):
		pl.savefig(filename, bbox_inches='tight')
	else:
		pl.show()
	pl.clf()

from numpy import array, array_equal, allclose
def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if array_equal(elem, myarr)), False)

def arreq_in_list_special(myarr,list_arrays):
	for elem in list_arrays:
		if array_equal(elem,myarr):
			return True
		if set(elem).issuperset(set(myarr)):
			return True
		if set(myarr).issuperset(set(elem)):
			return True
	return False

## generateMissingValuesNew
# @param: Dataset, (records,attributes)
# @param: k, integer that defines the number of monotone-mixture patterns generated
# @param: Mechanism: String, either "MAR", "MCAR" or "MNAR", at the moment only MAR is supported.
# @param: Perc: float, percentage of records with missing values
# 
# @param: AttrMixtureMin: integer, minimum amount of attributes in one univariate pattern, None means no minimum
# @param: AttrMixtureMax: integer, maximum amount of attributes in one univariate pattern, None means no maximum
# 
# @param: MixtureSizeMin: integer, Minimum size of each mixture pattern (amount of univariate patterns), None means no minimum
# @param: MixtureSizeMax: integer, Maximum size of each mixture pattern (amount of univariate patterns), None means no maximumx
# @param: MixtureDistribution: String, Distribution of records over all mixtures, can be "binomial","random" or "power"
# @param: PatternDistribution: String, Distribution of records each pattern inside the mixtures, can be "binomial", "random" or "power"
# 
# @param: RecordDistribution, String or float, Distribution of records with missing values per mixture, can be binomial, random or a float value to mix both
def generateMissingValuesNew(Dataset, k=3, Mechanism="MAR", Perc=20.0, AttrMixtureMin=None, AttrMixtureMax=None, MixtureSizeMin=None, MixtureSizeMax=None, MixtureDistribution="binomial", PatternDistribution="binomial", RecordDistribution=0.3):

	N = len(Dataset)
	M = len(Dataset[0])

	#convert the None parameters to their max and min
	if (AttrMixtureMin == None):
		AttrMixtureMin = 0
	if (AttrMixtureMax == None):
		AttrMixtureMax = M -1 #amount of attributes -1
	if (MixtureSizeMin == None):
		MixtureSizeMin = 0
	if (MixtureSizeMax == None):
		MixtureSizeMax = M -1 #amount of attributes -1 so that we can not get one mixture that takes all attributes

	MaxRecordsWithMissingValues = int(N * (Perc/100.0))

	#first generate a number of mixtures using k and the 
	mixtures = []
	allpatterns = []
	for i in range(k):
		patterns = []
		mixturesize = max(np.random.randint(low=MixtureSizeMin,high=MixtureSizeMax),1)
		#generate first patter
		AttrMixtureSize = max(np.random.randint(low=AttrMixtureMin,high=AttrMixtureMax),1) #minimum of 1
		pattern2 = np.arange(M)
		np.random.shuffle(pattern2)
		pattern = pattern2[:AttrMixtureSize]
		maxtries = 10
		while (arreq_in_list_special(pattern,allpatterns) and maxtries>0):
			maxtries -= 1
			np.random.shuffle(pattern2)
			pattern = pattern2[:AttrMixtureSize]
		patterns.append(pattern)
		allpatterns.append(pattern)
		#print len(pattern)
		#print "new pattern:",pattern
		tries = 100
		m = 0
		while m < mixturesize and tries > 0:
			tries -=1
			if len(pattern) == 1:
				break #is has no use to continue

			AttrMixtureSize = max(np.random.randint(low=max(AttrMixtureMin,len(pattern)/1.5),high=len(pattern) ),1) #minimum of 1
			np.random.shuffle(pattern)
			pattern_temp = copy.deepcopy(pattern)
			pattern_temp = pattern_temp[:AttrMixtureSize] #take subpattern
			#print "all",allpatterns,"pattern",pattern_temp
			if (not arreq_in_list(pattern_temp,allpatterns)):
				tries = 100
				m +=1
				pattern = pattern_temp
				#print len(pattern)
				#print "new pattern:",pattern
				allpatterns.append(pattern)
				patterns.append(pattern)
		c_patterns = copy.deepcopy(patterns)
		mixtures.append(c_patterns)
	#print mixtures

	#now that we have the mixtures, lets continue to the real job, deleting values
	RowsToProcess = np.arange(N)
	np.random.shuffle(RowsToProcess)
	RowsToProcess = RowsToProcess[:MaxRecordsWithMissingValues]
	processedRows = []
	processed = 0
	while processed < MaxRecordsWithMissingValues:

		#first check which mixture we will apply
		if MixtureDistribution == "random":
			currentMixtureIndex = np.random.randint(len(mixtures)) #choose a random mixture
		elif MixtureDistribution == "binomial":
			currentMixtureIndex = np.random.binomial(len(mixtures)-1,0.5)
		elif MixtureDistribution == "power":
			currentMixtureIndex = int(np.random.power(5.)*(len(mixtures)-1) )
		else: #error but lets just take random
			print "Warning: MixtureDistribution should be either \"random\", \"binomial\" or \"power\"."
			currentMixtureIndex = np.random.randint(len(mixtures)) #choose a random mixture

		#print "mixtureindex",currentMixtureIndex,"mixture",mixtures
		currentMixture = mixtures[currentMixtureIndex]

		## now get a row to apply a pattern on
		if (RecordDistribution == "random"):
			row = RowsToProcess[processed]
		elif (RecordDistribution == "binomial"):
			row = int(np.random.binomial(N-1,((currentMixtureIndex+1.0)/(len(mixtures)+1.0 ))))
			while (row in processedRows):
				row += 1
				if (row > N-1):
					row = 0
		else: #mix between the two
			if (not isinstance( RecordDistribution, float )):
				RecordDistribution = 0.3
				print "Warning: RecordDistribution should be either \"random\",\"binomial\" or a float value that specifies the ratio of binomial/random"
			if np.random.rand() > RecordDistribution:
				row = RowsToProcess[processed]
			else:
				row = int(np.random.binomial(N-1,((currentMixtureIndex+1.0)/(len(mixtures)+1.0 ))))
			while (row in processedRows):
				row += 1
				if (row > N-1):
					row = 0



		processedRows.append(row)
		processed += 1

		#now check which exact pattern to apply
		if PatternDistribution == "random":
			PatternIndex = np.random.randint(len(mixtures)) #choose a random mixture
		elif PatternDistribution == "binomial":
			PatternIndex = np.random.binomial(len(currentMixture)-1,0.5)
		elif PatternDistribution == "power":
			PatternIndex = int(np.random.power(5.)*(len(currentMixture)-1) )
		else: #error but lets just take random
			print "Warning: PatternDistribution should be either \"random\", \"binomial\" or \"power\"."
		
		#print "patternindex",PatternIndex,"len mixture",len(currentMixture
		if (PatternIndex > len(currentMixture)-1):
			PatternIndex = 0
		currentPattern = currentMixture[PatternIndex]
		for p in currentPattern:
			Dataset[row,p] = np.nan; #delete value

	return Dataset
'''

from functools import reduce
import operator


#cleveland bruteforce test
d,regression,Dim,amount,missing = GetDataSet(dataname='horse-colic')
all_mixtures,all_mixture_records, all_num_mixture_records = analyzeDatasetBruteForce(d.data)

print "option & length & mixt & patterns & records & inner mixt & inner rec "
for mi in range(len(all_mixtures)):
	mixtures = all_mixtures[mi]
	mixture_records = all_mixture_records[mi]
	num_mixture_records = all_num_mixture_records[mi]
	
	normalfactor = sum(num_mixture_records)/float(len(num_mixture_records))
	normalizednum = [x/float(normalfactor) for x in num_mixture_records]
	inner2 = reduce(operator.mul,normalizednum,1)
	lengths_mixt = [len(i) for i in mixtures]
	normalfactor = sum(lengths_mixt)/float(len(lengths_mixt))
	normalizedmixt = [x/float(normalfactor) for x in lengths_mixt] 
	inner1 = reduce(operator.mul,normalizedmixt,1)

	print mi,"&",len(mixtures),"&",max(lengths_mixt),"&",max(num_mixture_records),"&",inner1,"&",inner2




exit()

#small test
for dname in []:#"concrete"]:# ["wiki","automobile","post-operative","wisconsin","marketing","dermatology","mammographic","horse-colic","hepatitis","cleveland","bands", "adult","census"]:
	for k in range(1,5):
		for method in ["random","binomial","power"]:
			#PatternDistribution="binomial", RecordDistribution="random"
			for pattern in ["random","binomial","power"]:
				for record in ["random","binomial",0.3]:
	
					d,regression,Dim,amount,missing = GetDataSet(dataname=dname)
					d.data = generateMissingValuesNew(d.data,k=k,MixtureDistribution=method,PatternDistribution=pattern, RecordDistribution=record )

					#print d.data.shape
					mixtures,records, numbers = analyzeDataset(d.data)

					#for i in range(len(mixtures)):
						#print "Mixture", mixtures[i], "len records:", numbers[i]
					numbers = np.array(numbers)
					percent_numbers = numbers.astype(float) / np.sum(numbers)
					totoal_perc =  np.sum(numbers) / float( len(d.data) )

					print dname, "&",len(mixtures),"&", percent_numbers, "&", totoal_perc, "\\\\"

					if (record == 0.3):
						record = "mix"
					plotPatterns(records,'concrete img/'+method+"_"+pattern+"_"+record+`k`+dname+'_mixtures.png',True)
					plotPatternsInRecords(mixtures,records,'concrete img/'+method+"_"+pattern+"_"+record+`k`+dname+'_mapping.png',True)
		
for dname in ["wiki","automobile","post-operative","wisconsin","marketing","dermatology","mammographic","horse-colic","hepatitis","cleveland","bands", "adult","census"]:
	d,regression,Dim,amount,missing = GetDataSet(dataname=dname)
	#d.data = generateMissingValuesNew(d.data,k=k,MixtureDistribution=method,PatternDistribution=pattern, RecordDistribution=record )

	#print d.data.shape
	mixtures,records, numbers = analyzeDataset(d.data)
	#for i in range(len(mixtures)):
		#print "Mixture", mixtures[i], "len records:", numbers[i]
	numbers = np.array(numbers)
	percent_numbers = numbers.astype(float) / np.sum(numbers)
	totoal_perc =  np.sum(numbers) / float( len(d.data) )

	print dname, "&",len(mixtures),"&", percent_numbers, "&", totoal_perc, "\\\\"

	#if (record == 0.3):
	#	record = "mix"
	plotPatterns(records,'analysis img/'+dname+'_mixtures4.png',True,4)
	plotPatternsInRecords(mixtures,records,'analysis img/'+dname+'_mapping4.png',True,4)
	exit()
'''