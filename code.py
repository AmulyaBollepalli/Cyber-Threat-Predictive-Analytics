import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE


train=pd.read_csv('NSL_Dataset/Train.txt',sep=',')
test=pd.read_csv('NSL_Dataset/Test.txt',sep=',')

train.head()

columns=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
         "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations", 
         "num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate",
         "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
         "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
         "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
         "dst_host_srv_rerror_rate","attack","last_flag"] 

len(columns)

train.columns=columns
test.columns=columns

train.head()

test.head()

train.info()

test.info()

data = train.append(test, ignore_index=True)

# Counting the frequency of each element
unique_elements, counts = np.unique(data['attack'], return_counts=True)

# Displaying the frequency of each element
for element, count in zip(unique_elements, counts):
    print(f"Element: {element}, Frequency: {count}")
    
print(len(unique_elements))


data.loc[data.attack=='normal','attack_class']=0

data.loc[(data.attack=='back') | (data.attack=='land') | (data.attack=='pod') | (data.attack=='neptune') | 
         (data.attack=='smurf') | (data.attack=='teardrop') | (data.attack=='apache2') | (data.attack=='udpstorm') | 
         (data.attack=='processtable') | (data.attack=='worm') | (data.attack=='mailbomb'),'attack_class']=1

data.loc[(data.attack=='satan') | (data.attack=='ipsweep') | (data.attack=='nmap') | (data.attack=='portsweep') | 
          (data.attack=='mscan') | (data.attack=='saint'),'attack_class']=2

data.loc[(data.attack=='guess_passwd') | (data.attack=='ftp_write') | (data.attack=='imap') | (data.attack=='phf') | 
          (data.attack=='multihop') | (data.attack=='warezmaster') | (data.attack=='warezclient') | (data.attack=='spy') | 
          (data.attack=='xlock') | (data.attack=='xsnoop') | (data.attack=='snmpguess') | (data.attack=='snmpgetattack') | 
          (data.attack=='httptunnel') | (data.attack=='sendmail') | (data.attack=='named'),'attack_class']=3

data.loc[(data.attack=='buffer_overflow') | (data.attack=='loadmodule') | (data.attack=='rootkit') | (data.attack=='perl') | 
          (data.attack=='sqlattack') | (data.attack=='xterm') | (data.attack=='ps'),'attack_class']=4



data.head()

data.info()

# Counting the frequency of each element
unique_elements, counts = np.unique(data['attack_class'], return_counts=True)

# Displaying the frequency of each element
for element, count in zip(unique_elements, counts):
    print(f"Element: {element}, Frequency: {count}")
    
print(len(unique_elements))

data.isnull().sum()

data.drop(columns=['attack'], inplace=True)

data.head()

data.shape

# Handling Duplicate Values
data.drop_duplicates(inplace=True)  # Dropping duplicates

# Handling Categorical Values
categorical_cols = ['protocol_type', 'service', 'flag']
for col in categorical_cols:
    label_encoder = LabelEncoder()
    data[col] = label_encoder.fit_transform(data[col])
    mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    print(mapping)

data.shape

# Outlier Detection
clf = IsolationForest(random_state=42)
outliers = clf.fit_predict(data.drop('attack_class', axis=1))
data['outlier'] = outliers  # Adding outlier flag
data = data[data['outlier'] == 1]  # Removing outliers

data.head()

len(data.columns)

data.shape

# Feature Selection
X = data.drop(['attack_class', 'outlier'], axis=1)
y = data['attack_class']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)
feature_importances = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['importance'])
selected_features = feature_importances[feature_importances['importance'] > 0.03].index.tolist()
selected_features.append("attack_class")
data = data[selected_features]

len(data.columns)

data.columns

data.head()

data

data.to_csv('preprocessed.csv',index=False)

X = data.drop(['attack_class'], axis=1)
y = data['attack_class']

# Counting the frequency of each element
unique_elements, counts = np.unique(y, return_counts=True)

# Displaying the frequency of each element
for element, count in zip(unique_elements, counts):
    print(f"Element: {element}, Frequency: {count}")
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

X_test_Original=X_test

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Data Visualization
# You can visualize the distribution of attack types
sns.countplot(y)
plt.title('Distribution of Attack Types')
plt.xlabel('Attack Type')
plt.ylabel('Frequency')
plt.show()

accuracy_values = {}  # Dictionary to store accuracy values for all models

# Train and evaluate classifiers
classifiers = {
    "Decision Tree": DecisionTreeClassifier(),
    "LogisticRegression":LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier()
}

from sklearn.pipeline import Pipeline
import pickle

outputs=[]

for clf_name, clf in classifiers.items():
    print(f"\n{clf_name}:")
    # Train the classifier
    clf.fit(X_train, y_train)
    # Predict
    y_pred = clf.predict(X_test)
    # Performance Metrics
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_values[clf_name] = accuracy  # Store accuracy value
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", classification_rep)
    print("ROC AUC:", roc_auc)
    
    # Create a pipeline that first scales the data, then trains the model
    pipeline = Pipeline([('scaler', scaler), (clf_name, model)])

    # Train the pipeline (scaling + model training)
    pipeline.fit(X_train, y_train)

    # Save the pipeline (scaler + trained model) to a file using pickle
    with open(clf_name+'.pkl', 'wb') as file:
        pickle.dump(pipeline, file)
        
    
    if clf_name=="XGBoost":
        outputs=y_pred

# Plotting the bar chart for model comparison
plt.figure(figsize=(10, 6))
plt.bar(accuracy_values.keys(), accuracy_values.values(), color=['blue', 'orange', 'green', 'red'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracy')
plt.ylim(0, 1)  # Set y-axis limit to 0-1 for accuracy
plt.xticks(rotation=45)
plt.show()


