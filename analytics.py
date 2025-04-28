import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.stats import pearsonr  # <-- NEW for individual correlations

# Load Data
df = pd.read_csv("MDRPSurveyResults.csv")


# Pie Charts for Demographic Variables
def plot_pie_charts(df, columns):
    for col in columns:
        plt.figure()
        df[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=140, textprops={'fontsize': 12})
        plt.title(f"Distribution of {col}", fontsize=14)
        plt.ylabel('')
        plt.tight_layout()
        plt.show()


demographic_cols = ['AGE_RANGE', 'GENDER', 'CHRISTIAN_DURATION']
other_cols = ['CHURCH_ATTENDANCE', 'PERSONAL_PRAYER', 'BIBLE_READING', 'SIN_BEHAVIOUR', 'SIN_TEMPTATION', 'SIN_ENVIRONMENT',
                                'SIN_RESISTANCE', 'SIN_AWARENESS']
plot_pie_charts(df, demographic_cols)
plot_pie_charts(df, other_cols)

# Scores and Correlation Analysis
# Higher Resistance to Sin implies a Lower Score for Sin
df['SIN_RESISTANCE_ADJUSTED'] = 6 - df['SIN_RESISTANCE']

# Scoring Metrics
df['RELIGIOSITY_SCORE'] = df[['CHURCH_ATTENDANCE', 'PERSONAL_PRAYER', 'BIBLE_READING']].sum(axis=1)
df['PERCEIVED_SIN_SCORE'] = df[['SIN_BEHAVIOUR', 'SIN_TEMPTATION', 'SIN_ENVIRONMENT',
                                'SIN_RESISTANCE_ADJUSTED', 'SIN_AWARENESS']].sum(axis=1)

# Correlation Matrix (Religiosity vs Perceived Sin)
correlation_matrix = df[['RELIGIOSITY_SCORE', 'PERCEIVED_SIN_SCORE']].corr()
print("\nCorrelation Matrix (Religiosity vs Perceived Sin Score):\n")
print(correlation_matrix)

# Visualizing Overall Correlation
plt.figure()
sns.scatterplot(x='RELIGIOSITY_SCORE', y='PERCEIVED_SIN_SCORE', data=df)
sns.regplot(x='RELIGIOSITY_SCORE', y='PERCEIVED_SIN_SCORE', data=df, scatter=False, color='red')
plt.title("Correlation between Religiosity and Perceived Sin")
plt.xlabel("Religiosity Score")
plt.ylabel("Perceived Sin Score")
plt.tight_layout()
plt.show()

# Basic EDA for Variables
# Summary statistics
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

#Individual Correlations

print("\nIndividual Correlation Analysis:")

# Define sin-related variables
sin_variables = {
    'SIN_BEHAVIOUR': 'Sin Engagement',
    'SIN_TEMPTATION': 'Sin Temptation',
    'SIN_ENVIRONMENT': 'Sinful Environment',
    'SIN_RESISTANCE_ADJUSTED': 'Sin Resistance (Adjusted)',
    'SIN_AWARENESS': 'Sin Awareness'
}

# Loop through each sin variable
for var, label in sin_variables.items():
    corr, pval = pearsonr(df['RELIGIOSITY_SCORE'], df[var])
    print(f"\nCorrelation between Religiosity Score and {label}:")
    print(f"r = {corr:.2f}, p-value = {pval:.4f}")

    # Scatterplot for each relationship
    plt.figure()
    sns.scatterplot(x='RELIGIOSITY_SCORE', y=var, data=df)
    sns.regplot(x='RELIGIOSITY_SCORE', y=var, data=df, scatter=False, color='red')
    plt.title(f"Religiosity vs {label}")
    plt.xlabel("Religiosity Score")
    plt.ylabel(label)
    plt.tight_layout()
    plt.show()
