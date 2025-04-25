import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

df = pd.read_csv("MDRPSurveyResults.csv")

#Pie Charts for Demographic Variables
def plot_pie_charts(df, columns):
    for col in columns:
        plt.figure()
        df[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=140, textprops={'fontsize': 12})
        plt.title(f"Distribution of {col}", fontsize=14)
        plt.ylabel('')
        plt.tight_layout()
        plt.show()

demographic_cols = ['AGE_RANGE', 'GENDER', 'CHRISTIAN_DURATION']
plot_pie_charts(df, demographic_cols)

#Scores and Correlation Analysis
# Higher Resistance to Sin implies a Lower Score for Sin
df['SIN_RESISTANCE_ADJUSTED'] = 6 - df['SIN_RESISTANCE']

# Scoring Metrics
df['RELIGIOSITY_SCORE'] = df[['CHURCH_ATTENDANCE', 'PERSONAL_PRAYER', 'BIBLE_READING']].sum(axis=1)
df['PERCEIVED_SIN_SCORE'] = df[['SIN_BEHAVIOUR', 'SIN_TEMPTATION', 'SIN_ENVIRONMENT',
                                'SIN_RESISTANCE_ADJUSTED', 'SIN_AWARENESS']].sum(axis=1)

# Correlation Matrix (Religiosity Vs Perceived Sin)
correlation_matrix = df[['RELIGIOSITY_SCORE', 'PERCEIVED_SIN_SCORE']].corr()
print(correlation_matrix)

#Visualizing Correlation
plt.figure()
sns.scatterplot(x='RELIGIOSITY_SCORE', y='PERCEIVED_SIN_SCORE', data=df)
sns.regplot(x='RELIGIOSITY_SCORE', y='PERCEIVED_SIN_SCORE', data=df, scatter=False, color='red')
plt.title("Correlation between Religiosity and Perceived Sin")
plt.xlabel("Religiosity Score")
plt.ylabel("Perceived Sin Score")
plt.tight_layout()
plt.show()

#Basic EDA for Variables:
# Summary statistics
# Select only numeric columns for summary
numeric_cols = ['CHURCH_ATTENDANCE', 'PERSONAL_PRAYER', 'BIBLE_READING',
                'SIN_BEHAVIOUR', 'SIN_TEMPTATION', 'SIN_ENVIRONMENT',
                'SIN_RESISTANCE', 'SIN_AWARENESS',
                'RELIGIOSITY_SCORE', 'PERCEIVED_SIN_SCORE']

# Describe numeric variables
eda_summary = df[numeric_cols].describe().T  # Transpose for better formatting
eda_summary = eda_summary[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].round(2)

# Print as a visual table
print("\nExploratory Data Analysis Summary:\n")
print(tabulate(eda_summary, headers='keys', tablefmt='fancy_grid'))


