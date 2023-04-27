from matplotlib import pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_excel('C:/Users/08015437113/Documents/aprendmaq/ml-2023-1-trabalho-2/ml-2023-1-trabalho-1/dataset.xlsx')

data = data.dropna(axis=1, how='all')

for y in data.columns:
    if data[y].dtype == 'object': 
        lbl = LabelEncoder()
        lbl.fit(list(data[y].values))
        data[y] = lbl.transform(list(data[y].values))

colunas = ['Patient ID', 'Patient age quantile', 'SARS-Cov-2 exam result',
           'Patient addmited to regular ward (1=yes, 0=no)',
           'Patient addmited to semi-intensive unit (1=yes, 0=no)',
           'Patient addmited to intensive care unit (1=yes, 0=no)']

dt_copy = data.copy()
dt_copy.drop(colunas, axis=1, inplace=True)

for column in dt_copy.columns:

    q1 = dt_copy[column].quantile(0.25)
    q3 = dt_copy[column].quantile(0.75)
    iqr = q3 - q1

    li = q1 - (1.5 * iqr)
    ls = q3 + (1.5 * iqr) 

    outliers = (dt_copy[column] < li) | (dt_copy[column] > ls)

    dt_copy.loc[outliers, column] = np.nan

dt_copy = dt_copy.join(data['SARS-Cov-2 exam result'])

dt_copy = dt_copy[dt_copy.columns[dt_copy.isna().sum()/dt_copy.shape[0] < 0.9]]
dt_copy = dt_copy.fillna(dt_copy.median())

# exibindo correlações
correlations = dt_copy.corrwith(dt_copy['SARS-Cov-2 exam result'])
print(correlations.sort_values(ascending=False).head(10))
print('\n\n')
print(correlations.sort_values(ascending=True).head(10))
print('\n\n')
print(correlations.abs().sort_values(ascending=False).head(10))
print('\n\n')

# criando listas X (features) e y (resultado esperado)
X = dt_copy.drop(columns=['SARS-Cov-2 exam result'])

X = dt_copy[['Mean platelet volume ', 'Platelets']]

scaler =StandardScaler().fit(X)
X = scaler.transform(X)

y = dt_copy['SARS-Cov-2 exam result']

# separando 66% do dataset para treinamento e 33% para validação
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=101)

names = [
    "Nearest Neighbors",
    "Decision Tree",
    "Naive Bayes",
]

classifiers = [
    KNeighborsClassifier(n_neighbors=3),
    DecisionTreeClassifier(),
    GaussianNB()
]

results = []

for model_names, model in zip(names,classifiers):
    
    model.fit(X_train, y_train)
    #model = LogisticRegression().fit(X_train, y_train)
    
    prds = model.predict(X_test)
    
    cn = confusion_matrix(y_test, prds).ravel()
    
    tn, fp, fn, tp = confusion_matrix(y_test, prds).ravel()
    acs = accuracy_score(y_test, prds)
     
    print(f'Model {model_names}:\n\n'
          f'tn {tn}, fp {fp}, fn {fn}, tp {tp}\n\n'
          f'Accuracy: {acs}\n\n'
      f'Classification Report:\n{classification_report(y_test, prds)}\n')
    
    disp = DecisionBoundaryDisplay.from_estimator(
    model, X_train, response_method="predict",
    xlabel="Mean platelet volume", ylabel="Platelets",
    alpha=0.5,
    
    )
    
    disp.ax_.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor="k")
    plt.title(f'{model_names}')
    plt.axis('tight')

plt.show()