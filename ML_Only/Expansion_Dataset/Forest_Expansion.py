from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import pickle

data = pd.read_excel("Expansions_Fixed.xlsx")
data_teste = pd.read_excel("Expansions_Fixed_Teste.xlsx")

X_train = data.iloc[:, 0:-2]
y_train = data.iloc[:, -1]

X_test = data_teste.iloc[:, 0:-2]
y_test = data_teste.iloc[:, -1]


edge_encoder = preprocessing.LabelEncoder()
X_train['Edge'] = edge_encoder.fit_transform(X_train['Edge'])
pos_encoder = preprocessing.LabelEncoder()
X_train['Pos'] = pos_encoder.fit_transform(X_train['Pos'])
poscoarse_encoder = preprocessing.LabelEncoder()
X_train['PosCoarse'] = poscoarse_encoder.fit_transform(X_train['PosCoarse'])

X_test['Edge'] = edge_encoder.fit_transform(X_test['Edge'])
X_test['Pos'] = pos_encoder.fit_transform(X_test['Pos'])
X_test['PosCoarse'] = poscoarse_encoder.fit_transform(X_test['PosCoarse'])

edge_encoder_saving = open('edge_encoder.pkl', 'wb')
pickle.dump(edge_encoder, edge_encoder_saving)
edge_encoder_saving.close()
pos_encoder_saving = open('pos_encoder.pkl', 'wb')
pickle.dump(pos_encoder, pos_encoder_saving)
pos_encoder_saving.close()
poscoarse_encoder_saving = open('poscoarse_encoder.pkl', 'wb')
pickle.dump(poscoarse_encoder, poscoarse_encoder_saving)
poscoarse_encoder_saving.close()

n_estimators = [*range(50, 200, 5)]
criterion = ['gini', 'entropy', 'log_loss']
max_feature = ['sqrt', 'log2']
min_samples_split = [*range(2, 6, 1)]
class_weight = ['balanced', 'balanced_subsample']

best_accuracy = 0
for estimator in (pbar := tqdm(n_estimators)):
    for criteria in criterion:
        for feature in max_feature:
            for samples_split in min_samples_split:
                for weight in class_weight:
                    rf = RandomForestClassifier(n_estimators=estimator, criterion=criteria, max_features=feature, min_samples_split=samples_split, class_weight=weight)
                    rf.fit(X_train, y_train)
                    y_pred = rf.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_precision = precision
                        best_recall = recall
                        best_f1 = f1
                        best_params = [estimator, criteria, feature]
                        with open('BestRF_v2.pkl', 'wb') as f:
                            pickle.dump(rf, f)

                    pbar.set_description(f"Current Accuracy {accuracy:.2f} Best: {best_accuracy:.2f}")

print(best_accuracy, best_f1, best_precision, best_recall)
print(best_params)