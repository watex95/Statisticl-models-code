#DATA WRANGLING AND EDA
#################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

exp_df=pd.read_csv('/content/expected_ctc.csv')
exp_df.head()

#Drop ID columns
exp_df=exp_df.drop(['IDX','Applicant_ID'],axis=1)


#Lets get the summary statistics of the numeric variables
""".
exp_df.describe()
sns.set_palette("GnBu_d")
sns.set_style('whitegrid')
# Total experience correlated with expected CTC.
sns.jointplot(x='Total_Experience',y='Expected_CTC',data=exp_df)


"""Let's explore these types of relationships across the entire data set. Use pairplot to recreate the plot below.(Don't worry about the the colors)"""
sns.pairplot(exp_df)

"""Add heatmap to clearly see the correlation"""

#Heat map plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 6))
corr = exp_df.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);




















