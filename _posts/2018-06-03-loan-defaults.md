---
title: "Lending Club Loan Defaults"
date: 2018-06-03
tags: [loan defaults]
header:
  image: "/images/Lending Club/Lending_Club.png"
excerpt: "A project to predict Lending Club loan defaults"  
mathjax: "true"
---

# The Problem

For my General Assembly Data Science final project, I looked at loan defaults. Loan defaults can be extremely problematic and not only lead to financial loss for the lending organisation but also to long term financial issues for the defaulting borrower. The default will be listed on the borrowers credit report. This will make it harder for the borrower to take out future loans or mortgages as well as open credit accounts because financial organisations will be suspicious that he may not be able to meet his debt repayments. Furthermore, defaulting borrowers will have the issue of increasing debt due to interest and fees.

Knowing if a prospective borrower is likely to default will allow the organisation to reject the application and, therefore, prevent any losses from a default. This would also ensure that the prospective borrower could not default and suffer financially from it.

With this in mind, I had two main goals with the project:

- To build a machine learning model to predict loan defaults.
- To use the model to gain insights into the main factors behind the defaults.


# Lending Club

I sourced my data from Lending Club, a US-based peer-to-peer lending platform and the largest of its kind in the world. I downloaded loan data from their [website](https://www.lendingclub.com/info/download-data.action), covering a 10 year period from 2007 to 2017. The whole dataset included data on nearly 1.8 million loans and was split into 12 different csv files.

Due to the size of the data, and the fact that it was split across multiple files, I decided to put all of the data into a Postgresql database. This would allow me to join all of the separate tables into one large table containing all records as well as allowing me to use SQL’s queries and joins throughout the project.

The dataset contained 145 features and these appeared to be split into two groups: features about the loan; features about the borrower. The features about the loan included loan amount, interest rate, loan grade, issue date, etc. The features about the borrower included annual income, DTI (debt-to-income ratio), number of bank accounts, previous delinquencies, etc.


# US Census Data

One of the features of the loan dataset was the borrowers’ zip code, i.e. where the borrower lived. I thought that it would be interesting to see if I could map macroeconomic data to my loan data based on the zip codes. I wanted to see if perhaps the affluence of the area that someone lives in could be a predictor for loan defaults, e.g. perhaps less affluent areas would see higher default rates. I was able to download data containing median household income per zip code from the US Census Bureau. However, there were some limitations with my approach. I was only able to download census data from 2011 to 2016. My loan data covered the years 2007 to 2017. This meant that I could not map the census data to loan data based on year. Therefore I had to average the median household income which generalised it. Another limitation was the fact that my loan data only contained the first three digits of the zip code which represents a very large area. Therefore, I had to further average out the median household incomes for each three digit zip code which lead to more generalisation of this macroeconomic factor.


# Exploratory Data Analysis

During exploratory data analysis (EDA), the data required extensive cleaning. There were many columns with missing values, numeric features that needed symbols to be removed (e.g. ‘$’ or ‘%’), and date related columns that needed to be converted to datetime format.

```python
df.int_rate = df.int_rate.str.replace('%', '')
```

```python
loan.issue_d = pd.to_datetime(loan.issue_d)
```

The target column was loan status which lists the current status of the loan. Values included ‘Current’, ‘Fully Paid’, ‘Default’, ‘Charged Off’, and various stages of ‘Late’. As the goal of the project was to predict defaults, I would only be looking at the loans that had come to completion and had either been fully paid or defaulted. I would not be using the loans that were current or late because I did not know what the end result of those loans would be.

Lending Club defines defines 'Default' as 120+ days past due. 'Charged Off' is defined as 150+ days past due with no expectation of the loan being paid. For the purpose of this project, I grouped these two categories together as ‘Default'. The prediction would therefore be a simple binary classification between ‘Fully Paid’ (0) or ‘Default’ (1).

```python
loan.loan_status = loan.loan_status.map(lambda x: 'Default' if x=='Charged Off' else x)
```
```python
loan.loan_status = loan.loan_status.map(lambda x: 1 if x=='Default' else 0)
```

I had to analyse every feature in the dataset to make sure that it was relevant to the problem. There were many features related to what happens after a loan has completed such as payment plans and collections for defaulted loans. Therefore, these had to be dropped. There were many other columns that were either redundant or did not contain data covering the whole 10 year period.

When analysing the data, one can see the growth of the company over the ten year period. In the wake of the 2008 financial crash, many people started to lose trust with the banks and move to other financial platforms such as peer-to-peer lenders. This is shown by the steady increase in number of loans and average loan amount year on year. This reached a peak in early 2016 at which point the company faced a few scandals and the number of loans started to drop.

<img src="{{ site.url }}{{ site.baseurl }}/images/Lending Club/loans_over_time.png" alt="number of loans & average loan amount over time">

When looking at the loans per state, it is no surprise to see that California has the largest total loan amount (nearly $4 billion) over the ten year period given that Lending Club is a San Francisco-based organisation. Other highly populated states such as New York, Florida, and Texas also had large total loan amounts of around $2 billion.

<img src="{{ site.url }}{{ site.baseurl }}/images/Lending Club/loans_per_state.png" alt="total loan amount per state">

I visualised all numerical features to look at their distributions, check for outliers, and also check for correlation. I removed correlated features to avoid multicollinearity in my models.

```python
# Create heatmap to show correlation between features.
corr = df.corr()
fig, ax = plt.subplots(figsize=(20,15))
mask = np.zeros_like(corr, dtype=np.bool)    # Create a mask to stop upper right portion of heatmap from showing.
mask[np.triu_indices_from(mask)] = True
cmap = (sns.color_palette("RdBu_r", 10))     # Diverging colour palette.
ax = sns.heatmap(corr, mask=mask, ax=ax, cmap=cmap, annot=True, linecolor ='white', linewidths=5)
ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=14)
plt.show()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/Lending Club/correlation.png" alt="correlation">

With the exploration, cleaning, and analysis complete, I could move on to the modelling.


# Modelling

Because most people do not default on their loans, one limitation I had from the start was a large class imbalance of around 80%, with most records being fully paid. This meant that I could not use accuracy score as a measure of the machine learning models’ performance. Instead, I would use other metrics to assess the performance of the models such as AUC-ROC and recall.

Models are never going to be 100% accurate and this is especially true with loan defaults when there are so many unknowns in the borrowers’ lives that could potentially have an impact on their finances. Therefore, any models will inevitably get some predictions wrong and either incorrectly predict a loan to default when it would end up actually fully paying (false positive) or incorrectly predict a loan to fully pay when it actually ends up defaulting (false negative). In the case of loans, it is preferable to have more false positives than false negatives in order to minimise the number of loans that actually end up defaulting and thereby creating a conservative model. It is for this reason that recall was the one of the primary metrics that I used to determine the models’ performances, where recall is defined as:

$$recall = \frac{true(+ve)}{true(+ve) + false(-ve)}$$

As the models require numerical input, I had to binarise all categorical variables, i.e. create dummy variables, whereby each class in a categorical feature would become its own binary feature. I then had to split the dataset into training and testing sets in order to train the model and then test it on new unseen data to measure its performance. As I had data covering a decade, I trained the model on pre-2017 data and tested it on 2017 data.

```python
df = pd.concat([df, pd.get_dummies(df.emp_length, drop_first=True, prefix='emp_length')], axis=1)
df.drop('emp_length', axis=1, inplace=True)
```

```python
df['year'] = df['issue_d'].dt.year
```
```python
# Training data is pre-2017.
train = df[df['year'] < 2017]
# Testing data is from 2017.
test = df[df['year'] == 2017]
# Now drop date columns.
train.drop(['issue_d', 'year'], axis=1, inplace=True)
test.drop(['issue_d', 'year'], axis=1, inplace=True)
```

The first machine learning algorithm that I used was logistic regression, a good model for predicting binary targets. However, this did not produce great results and did not predict many true positives (actual defaults) and had a recall score of 0.17 on the testing set. Therefore, I tried other classification algorithms, decision trees and random forest which saw no improvement. I used a gridsearch to try to find the optimal parameters for the algorithms but this did also not yield any improvements.

The problems with the predictions likely stemmed from the large class imbalance. To counteract this, I used resampling methods to increase the ratio of defaulted loans relative to fully paid loans in the training set. I used both upsampling (oversampling), to increase the number of defaulted loans, and downsampling (undersampling), to reduce the number of fully paid loans, to create training sets that had an approximate 60:40 ratio, majority fully paid.

```python
# Create dataframe for fully paid loans.
train0 = train[train.loan_status==0]
# Create dataframe for defaulted loans.
train1 = train[train.loan_status==1]
# Upsample defaults, approx 60:40 (majority fully paid)
upsample = resample(train1[train.loan_status == 1], replace = True, n_samples = 400000, random_state = 0)
# Concatonate fully paid loans and upsampled defaults.
train_up = pd.concat([train0, upsample], axis=0)
```

After resampling the training set, I ran logistic regression and random forest on both the upsampled and downsampled data. This resulted in the three best models of the project so far which are an upsampled logistic regression, a downsampled logistic regression, and a downsampled random forest. Of these three models, the downsampled logistic regression performs the best, having the highest number of true positives and the fewest false negatives. Both the downsampled logistic regression and random forest had a recall score of 0.51 for defaults but the logistic regression had a higher AUC on the ROC curve, 0.70 compared to 0.67.

<img src="{{ site.url }}{{ site.baseurl }}/images/Lending Club/AUC_ROC.png" alt="AUC-ROC">

When looking at the coefficients and feature importances across the models, high interest rate was the single biggest factor in predicting defaults for all models. Other important features were high debt-to-income ratio (DTI) and also 60 month term for the logistic regression models.

<img src="{{ site.url }}{{ site.baseurl }}/images/Lending Club/coefficients.png" alt="coefficients">

The median household income data from the US Census did not end up being a strong predictor. This is likely due to the fact that the data was generalised too much to match the three digit zip code from the loan data.


# Conclusions

My results show that interest rate is the single biggest factor predicting loan defaults, with debt-to-income ratio (DTI) and term also being strong predictors. Defaulted loans have a higher average interest rate than fully paid loans. Defaulted loans also have a higher average DTI than fully paid loans. DTI is the ratio of the borrower’s monthly payments/debts to his monthly income. The term is the length of the repayment period and there are two options that a borrower can pick, either 36 months or 60 months. 60 month term has a much higher default rate than 36 month term.

<img src="{{ site.url }}{{ site.baseurl }}/images/Lending Club/int_rate.png" alt="interest rate" height="400" width="200">   <img src="{{ site.url }}{{ site.baseurl }}/images/Lending Club/DTI.png" alt="DTI" height="400" width="200">   <img src="{{ site.url }}{{ site.baseurl }}/images/Lending Club/term.png" alt="term" height="400" width="200">

The interest rates of the Lending Club loans are fixed and are calculated based on the loan grade (A1 - G5). A loan is assigned a grade by Lending Club based on the borrower’s credit report and any risk/volatility that the organisation detects. High grade loans (e.g. A) have a low interest rate whereas low grade loans (e.g. G) have a high interest rate. All loans have a base rate of 5.05% but the grade and sub-grade of the loan determines how much is added on to adjust for risk/volatility. Low grade G loans have a high interest rate above 30%.

<img src="{{ site.url }}{{ site.baseurl }}/images/Lending Club/Grades.png" alt="Lending Club grades">

The results of the modelling suggest that, in trying to account for risk in the loan, ironically, Lending Club is actually increasing the risk of default by assigning such a high interest rate to low grade loans. Borrowers with low grade loans who happen to have a high DTI are likely to have less disposable income available to repay the loan and less financial stability. A high interest loan would place them under significant financial pressure. If the borrower was to choose a 60 month term, then this would increase the pressure due to the fixed rate, high interest which would add up significantly over the term.

If I was to make business suggestions based on these results, it would be to reduce the interest rate for the low grade loans to reduce the financial pressure that these lower income borrowers face. Another option would be to simply scrap G grade loans and raise the threshold required to be granted a loan, thereby rejecting any prospective applicants who would have otherwise been assigned a G grade. This would reduce the number of defaulting loans. As Lending Club is a peer-to-peer lending platform, much of the risk/volatility adjustment is likely there to provide the investors with extra risk/reward for their investments. However, I do not believe that it is worth putting financially vulnerable, low grade borrowers under increased pressure for these potential investor gains.


# Next Steps

In its current state, improvements need to be made to increase the predictive power of the model. Some of the models that I ran likely had high variance from overfitting too many features in the training set. One solution could be to remove more features and reduce the overfitting. There are other algorithms that I could also run to see how they perform with this dataset. These include support vector machines, naive bayes, or even a neural network.

Despite the improvements that need to be made to the predictions I believe that I have gained some valuable insights into the main factors behind the Lending Club loan defaults.
