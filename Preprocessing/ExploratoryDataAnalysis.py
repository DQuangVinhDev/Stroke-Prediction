import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv ('./healthcare-dataset-stroke-data.csv', header =0, index_col="id")

def get_var_category(series):
    unique_count = series.nunique(dropna=False)
    total_count = len(series)
    if pd.api.types.is_numeric_dtype(series):
        return 'Numerical'
    elif pd.api.types.is_datetime64_dtype(series):
        return 'Date'
    elif unique_count==total_count:
        return 'Text (Unique)'
    else:
        return 'Nominal'

def print_categories(df):
  for column_name in df.columns:
          print(column_name, ": ", get_var_category(df[column_name]))

def find_outlier(df):
  q1=df.quantile(0.25)
  q3=df.quantile(0.75)

  iqr = q3 - q1
  outliers = df[((df < (q1 - 1.5 * iqr)) | ((df > (q3+1.5*iqr))))]
  print("number of outliers: " + str(len(outliers)))
  print("max outlier value: " + str(outliers.max()))
  print("min outlier value: " + str(outliers.min()))

def BMI():
    all_bmi = df['bmi']
    stroke_data = df[df['stroke'] == 1]
    stroke_bmi = stroke_data['bmi']
    plt.hist([all_bmi, stroke_bmi], 
			bins=50, 
			color=['lightblue', 'pink'],
			label=['All Cases', 'Stroke Cases'],
			stacked=True)
    plt.title('Distribution of BMI for Stroke Cases')
    plt.xlabel('BMI')
    plt.ylabel('Frequency')
    plt.show()
    print('bmi outliers: ')
    find_outlier(df['bmi'])
    print('\n')
    sns.boxplot(df['bmi'], color='lightblue')
    plt.xlabel('BMI')
    plt.ylabel('Values')
    plt.title('Box plot of average BMI')
    plt.show()

def glucose():
    all_glucose = df['avg_glucose_level']
    stroke_data = df[df['stroke'] == 1]
    stroke_glucose = stroke_data['avg_glucose_level']
    plt.hist([all_bmi, stroke_glucose], 
         bins=50, 
         color=['lightblue', 'pink'],
         label=['All Cases', 'Stroke Cases'],
         stacked=True)
    glucose_mean = df['avg_glucose_level'].mean()
    plt.axvline(x=glucose_mean, color='red',linestyle='dashed')
    plt.title('Distribution of Average glucose level for Stroke Cases')
    plt.xlabel('Glucose level')
    plt.ylabel('Frequency')
    plt.show()
    print('glucose level outliers: ')
    find_outlier(df['avg_glucose_level'])
    sns.boxplot(df['avg_glucose_level'], color='lightblue')
    plt.xlabel('Glucose level')
    plt.ylabel('Values')
    plt.title('Box plot of average glucose level')
    plt.show()

def age():
    all_age = df['age']
    stroke_data = df[df['stroke'] == 1]
    stroke_age = stroke_data['age']
    plt.hist([all_bmi, stroke_age], 
         bins=50, 
         color=['lightblue', 'pink'],
         label=['All Cases', 'Stroke Cases'],
         stacked=True)
    plt.title('Distribution of age for Stroke Cases')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()
    print('Age outliers: ')
    find_outlier(df['age'])
    sns.boxplot(df['age'], color='lightblue')
    plt.xlabel('Glucose level')
    plt.ylabel('Values')
    plt.title('Box plot of average glucose level')
    plt.show()

 
def numerical():
	df.replace('Unknown', np.nan, inplace=True)
	df.describe(exclude=[np.int64,np.object])
	numericalMenu()
	inp = input('\nEnter option ')
	while True:
		if inp == 'a':
			BMI()
		if inp == 'b':
			glucose()
		if inp == 'c':
			glucose()
		if inp == 'z':
			break
		print("\n")
		menu()
		inp = input('\nEnter option ')
			
def gender(): 
    ct = pd.crosstab(df['gender'], df['stroke'])
    ax = ct.plot(kind='bar', stacked=True, figsize=(8,6), alpha=0.7)
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.title('Stroke Cases by Gender')
    ax.legend(['Non-Stroke', 'Stroke'], loc='lower right')
    plt.show()
    gender_stroke = pd.crosstab(index=df['gender'], columns=df['stroke'], normalize='index') * 100
    gender_stroke = gender_stroke.round(2)
    print(gender_stroke)
    
def residence():
    ct = pd.crosstab(df['Residence_type'], df['stroke'])
    ax = ct.plot(kind='bar', stacked=True, figsize=(8,6), alpha=0.7)
    plt.xlabel('Residence Type')
    plt.ylabel('Count')
    plt.title('Stroke Cases by Residence Type')
    ax.legend(['Non-Stroke', 'Stroke'], loc='lower center')
    plt.show()
    residence_stroke = pd.crosstab(index=df['Residence_type'], columns=df['stroke'], normalize='index') * 100
    residence_stroke = residence_stroke.round(2)
    print(residence_stroke)

def hypertension():
    ct = pd.crosstab(df['hypertension'], df['stroke'])
    ax = ct.plot(kind='bar', stacked=True, figsize=(8,6), alpha=0.7)
    plt.xlabel('Hypertension')
    plt.ylabel('Count')
    plt.title('Stroke Cases by Hypertension')
    ax.legend(['Non-Stroke', 'Stroke'], loc='lower center')
    plt.show()
    hypertension_stroke = pd.crosstab(index=df['hypertension'], columns=df['stroke'], normalize='index') * 100
    hypertension_stroke = hypertension_stroke.round(2)
    print(hypertension_stroke)

def heart():
	ct = pd.crosstab(df['heart_disease'], df['stroke'])
	ax = ct.plot(kind='bar', stacked=True, figsize=(8,6), alpha=0.7)

	plt.xlabel('Heart disease')
	plt.ylabel('Count')
	plt.title('Stroke Cases by Heart Disease')
	ax.legend(['Non-Stroke', 'Stroke'], loc='lower center')

	plt.show()
	heart_disease_stroke = pd.crosstab(index=df['heart_disease'], columns=df['stroke'], normalize='index') * 100
	heart_disease_stroke = heart_disease_stroke.round(2)
	print(heart_disease_stroke)  

def married():
	ct = pd.crosstab(df['ever_married'], df['stroke'])
	ax = ct.plot(kind='bar', stacked=True, figsize=(8,6), alpha=0.7)

	plt.xlabel('Marriage Status')
	plt.ylabel('Count')
	plt.title('Stroke Cases by Marriage Status')
	ax.legend(['Non-Stroke', 'Stroke'], loc='lower center')

	plt.show()
	marriage_stroke = pd.crosstab(index=df['ever_married'], columns=df['stroke'], normalize='index') * 100
	marriage_stroke = marriage_stroke.round(2)
	print(marriage_stroke)
 
def work():
	ct = pd.crosstab(df['work_type'], df['stroke'])
	ax = ct.plot(kind='bar', stacked=True, figsize=(8,6), alpha=0.7)

	plt.xlabel('Work type')
	plt.ylabel('Count')
	plt.title('Stroke Cases by Work type')
	ax.legend(['Non-Stroke', 'Stroke'], loc='upper right')

	plt.show()
	work_stroke = pd.crosstab(index=df['work_type'], columns=df['stroke'], normalize='index') * 100
	work_stroke = work_stroke.round(2)
	print(work_stroke)
 
def categorical():
	categoricalMenu()
	inp = input('\nEnter option ')
	while True:
		if inp == 'a':
			gender()
		if inp == 'b':
			residence()
		if inp == 'c':
			hypertension()
		if inp == 'd':
			heart()
		if inp == 'e':
			married()
		if inp == 'f':
			work()
		if inp == 'z':
			break
		print("\n")
		menu()
		inp = input('\nEnter option ')
	

def correlation():
	df_copy = df.copy()
	ohe = OneHotEncoder()
	ss = StandardScaler()
	le = LabelEncoder()

	for col in df_copy.columns:
		df_copy[col] = le.fit_transform(df_copy[col])
		
	cols = df_copy.columns
	
	df_copy[cols] = ss.fit_transform(df_copy[cols])

	df_corr = df_copy.corr()
	mask = np.triu(np.ones_like(df_corr, dtype=np.bool))
	mask = mask[1:, :-1]
	corr = df_corr.iloc[1:,:-1].copy()

	sns.heatmap(corr, mask=mask, cmap='GnBu', annot=True, fmt='.2f')

def menu():
	print("Press 1 to print data information")
	print("Press 2 to print numeric data analysis")
	print("Press 3 to print categorical data analysis")
	print("Press 2 to print correlation data analysis")
	

def numericalMenu():
    print("Press a to print BMI analysis")
    print("Press b to print glucose level analysis")
    print("Press c to print age analysis")
    print("Print z to exit")
 
def categoricalMenu():
    print("Press a to print gender analysis")
    print("Press b to print residence analysis")
    print("Press c to print hypertension analysis")
    print("Press d to print heart analysis")
    print("Press e to print marrital status analysis")
    print("Press f to print work type analysis")
    print("Print z to exit")
    
 
menu()
val = input('\nEnter Number ')
val_int = int(val)

while True:
	if val_int == 1:
		df.info()
		print_categories(df)
		df.replace('Unknown', np.nan, inplace=True)
		df.isna().sum()
		df.duplicated().sum()
		st_plot = df["stroke"].value_counts().plot(kind="bar",color=["lightblue","pink"], rot=0)
		plt.xticks(ticks=[1,0], labels=["stroke", "no_stroke"]);
	if val_int == 2:
		numerical()
	if val_int == 3:
		categorical()
	if val_int == 4:
		correlation()
	if val_int == 0:
		break
	print("-------------END-------------")
	print("\n")
	menu()
	val = input('\nEnter Number ')
	val_int = int(val)

