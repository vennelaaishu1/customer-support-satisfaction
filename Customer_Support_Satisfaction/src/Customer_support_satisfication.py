# # Customer Support Ticket Analysis & Satisfaction Prediction
# 
# **Goal:** Explore a customer support ticket dataset and build classification models to predict `Customer Satisfaction Rating` (1-5) from ticket, customer, and product attributes — useful for flagging tickets at risk of a poor rating.

# ## 1. Setup

import warnings
warnings.filterwarnings("ignore")
import re
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# ## 2. Load the data
# 
# Uses a relative path so the notebook runs on any machine as long as `customer_support_dataset.csv` sits in the same folder.

#Reading the data
df = pd.read_csv("./data/customer_support_dataset.csv")

# ## 3. Initial exploration

df.head()

df.tail()

df.shape

df.columns

df.info()

df.describe()

df.isnull().sum()

#Checking duplicate line
duplicate = df.duplicated().any()
duplicate

df.info()
cat_cols  = df.select_dtypes(['object']).columns
int_cols  = df.select_dtypes(['int64']).columns
float_cols = df.select_dtypes(['float']).columns
print(cat_cols)
print(int_cols)
print(float_cols)

# **Findings so far:** 8,469 rows, 17 columns, no exact duplicate rows. Missing values only appear in `Resolution`, `First Response Time`, `Time to Resolution`, and `Customer Satisfaction Rating` — all for the same reason: those fields only get populated once a ticket is actually resolved/responded to.

# ## 4. Cleaning & feature engineering

#Convert to datetime
df['Date of Purchase'] = pd.to_datetime(df['Date of Purchase'], format = '%Y-%m-%d')
df['First Response Time'] = pd.to_datetime(df['First Response Time'], format = '%Y-%m-%d %H:%M:%S')
df['Time to Resolution'] = pd.to_datetime(df['Time to Resolution'], format = '%Y-%m-%d %H:%M:%S')

df1 = df.drop(columns = ['Ticket ID'])

df1['Resolution'] = df1['Resolution'].fillna('None')
df1['First Response Time'] = df1['First Response Time'].fillna('No response')
df1['Time to Resolution'] = df1['Time to Resolution'].fillna('No resolution')
df1['Customer Satisfaction Rating'] = df1['Customer Satisfaction Rating'].fillna('No rating')

df1.isnull().sum()

print(df1['Customer Age'].max())
print(df1['Customer Age'].min())

age = []
for i in df1['Customer Age']:
    if i<=30:
        age.append('Young Customer')
    elif 30<i<55:
        age.append('Middle Age Customer')
    else:
        age.append('Old Customer')
df1['Type of Customer'] = age

chart_age = df1['Type of Customer'].value_counts()
chart_gen = df1['Customer Gender'].value_counts()
plt.figure(figsize = (10,5))
plt.pie(chart_age, labels = chart_age.index, autopct='%.0f%%')
plt.title('Distribution of customer age', loc = 'left', pad = 10, size = 15)
plt.show()

unique = len(df1['Product Purchased'].unique())
print(f' Total of products was purchased: {unique}')

chance = []
for i in df1['First Response Time']:
    if i == 'No response':
        chance.append('No')
    else:
        chance.append('Yes')
df1['Response'] = chance

chart_res = df1['Response'].value_counts()
plt.pie(chart_res, labels = chart_res.index, autopct = '%.2f')
plt.title('Chance of response', loc = 'center', pad = 10, size = 15)
plt.show()

# ## 5. Choosing the prediction target: `Customer Satisfaction Rating`
# 
# `Customer Satisfaction Rating` (1-5) is only populated for **closed** tickets — 2,769 of the 8,469 rows. We filter down to just those rows, since an unrated ticket has no label to learn from.
# 
# Business case: flag tickets that are heading toward a low satisfaction score, so a team lead can step in before the ticket closes.

# Keep only tickets that actually have a satisfaction rating (i.e. closed tickets)
df1 = df1[df1['Customer Satisfaction Rating'] != 'No rating'].copy()
df1['Customer Satisfaction Rating'] = df1['Customer Satisfaction Rating'].astype(int)

print('Rows with a satisfaction rating:', df1.shape[0])
df1['Ticket Status'].value_counts()

rating_counts = df1['Customer Satisfaction Rating'].value_counts().sort_index()
print(rating_counts)

plt.figure(figsize=(6,4))
sns.barplot(x=rating_counts.index, y=rating_counts.values)
plt.title('Class balance: Customer Satisfaction Rating')
plt.xlabel('Rating')
plt.ylabel('Number of tickets')
plt.show()

# The five rating classes are reasonably balanced (~540-580 tickets each out of 2,769), so accuracy is still a trustworthy metric here. Note the much smaller sample size than the full dataset — fewer training examples per class, so results below are cross-validated (Section 12) to check they're not just noise from one lucky/unlucky split.
# 
# Since every remaining row is a `Closed` ticket, `Ticket Status` is now constant across the whole dataset and carries zero information — it will be dropped rather than encoded.

# ## 6. A quick NLP experiment: does the resolution text carry sentiment signal?
# 
# Before dropping the free-text `Resolution` field, let's actually test whether it holds useful signal, rather than assuming. We compute a simple lexicon-based sentiment score (count of positive support-words minus negative support-words, normalized by text length) for each resolution note, with no external NLP libraries required.

positive_words = {'good','great','excellent','happy','satisfied','resolved','fixed','helpful','quick',
                   'easy','love','thank','thanks','amazing','perfect','wonderful','best','pleasant',
                   'efficient','friendly','win','success','benefit','positive'}
negative_words = {'bad','terrible','angry','frustrated','disappointed','broken','issue','problem','delay',
                   'slow','poor','worst','annoyed','difficult','failed','error','complain','unhappy',
                   'waste','horrible','fail','negative','wrong'}

def sentiment_score(text):
    words = re.findall(r"[a-zA-Z']+", str(text).lower())
    if not words:
        return 0.0
    pos = sum(w in positive_words for w in words)
    neg = sum(w in negative_words for w in words)
    return (pos - neg) / len(words)

df1['Resolution Sentiment'] = df1['Resolution'].apply(sentiment_score)

print(df1['Resolution Sentiment'].describe())
print('Rows with a nonzero sentiment score:', (df1['Resolution Sentiment'] != 0).sum(), 'of', len(df1))
print('Correlation with Customer Satisfaction Rating:', df1['Resolution Sentiment'].corr(df1['Customer Satisfaction Rating']))

df1[['Resolution', 'Resolution Sentiment']].sample(5, random_state=3)

# **Result of the experiment:** correlation is essentially zero (~0.01), and only about 8% of resolution notes contain any sentiment word at all. Looking at the raw text confirms why: entries like *"Wish performance senior line however."* and *"Treat baby expect over discussion."* are grammatically-random sentences, not real support agent notes — there's no genuine sentiment to extract. We still include `Resolution Sentiment` as a feature below (it costs nothing and the model can simply learn to ignore it), but we don't expect it to move the needle, and the result below confirms that expectation rather than assuming it.

# ## 7. Preparing features for modeling

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler

df2 = df1.copy()

# Engineer simple numeric features from the purchase date
df2['Purchase_Month'] = df2['Date of Purchase'].dt.month
df2['Purchase_DayOfWeek'] = df2['Date of Purchase'].dt.dayofweek

# Drop free-text / identifier / raw-datetime / now-constant columns.
# Note: 'First Response Time' minus 'Time to Resolution' looks like it could make a useful
# 'resolution duration' feature, but checking it shows ~50% negative durations and a tiny
# +/-24h range -- these timestamps look randomly generated rather than sequential, so deriving
# a duration feature from them would just be encoding noise. Skipped for that reason.
drop_cols = ['Customer Name', 'Customer Email', 'Ticket Description', 'Ticket Subject',
             'Resolution', 'Date of Purchase', 'First Response Time', 'Time to Resolution',
             'Ticket Status']
df2 = df2.drop(columns=drop_cols)

df2.head()

# ### Encoding strategy
# 
# - **Target (`Customer Satisfaction Rating`)** -> already a clean integer 1-5, used directly as the class label.
# - **Low-cardinality nominal columns** (`Customer Gender`, `Ticket Type`, `Ticket Priority`, `Ticket Channel`, `Type of Customer`, `Response`) -> **one-hot encoded**. `Ticket Priority` is included as a feature here since it's known independently of the satisfaction outcome and may carry useful information.
# - **`Product Purchased`** (42 distinct values) -> **frequency-encoded**, recomputed on this filtered (closed-tickets-only) subset.
# - **`Resolution Sentiment`** -> already numeric from Section 6, used as-is.

# Frequency-encode the high-cardinality Product Purchased column (computed on this subset)
product_freq = df2['Product Purchased'].value_counts(normalize=True)
df2['Product Purchased Freq'] = df2['Product Purchased'].map(product_freq)

# One-hot encode the remaining low-cardinality nominal columns
onehot_cols = ['Customer Gender', 'Ticket Type', 'Ticket Priority', 'Ticket Channel',
               'Type of Customer', 'Response']
df2 = pd.get_dummies(df2, columns=onehot_cols, drop_first=False)

X = df2.drop(columns=['Customer Satisfaction Rating', 'Product Purchased'])
y = df2['Customer Satisfaction Rating']

# Keep a LabelEncoder around purely so we can print nice class names in reports below
target_le = LabelEncoder()
target_le.fit(y)

print('Feature matrix shape:', X.shape)
X.head()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numeric features (helps Logistic Regression and Gradient Descent converge properly)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print('Train shape:', X_train.shape, ' Test shape:', X_test.shape)

# ## 8. Model evaluation helper
# 
# Since `Customer Satisfaction Rating` is **ordinal** (a miss of 1 point, e.g. predicting `4` when the true value is `5`, is a smaller error than predicting `1`), plain accuracy alone understates near-misses. Alongside accuracy/precision/recall/F1, we add:
# - **MAE** (mean absolute error in rating points)
# - **Within ±1 accuracy** — how often the prediction is off by at most one point

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              classification_report, confusion_matrix, mean_absolute_error)

model_scores = []  # collects metrics for every model to compare at the end

def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    mae = mean_absolute_error(y_true, y_pred)
    within1 = (np.abs(np.array(y_true) - np.array(y_pred)) <= 1).mean()

    print(f'{name}')
    print(f'  Accuracy       : {acc:.4f}')
    print(f'  Precision      : {prec:.4f}  (weighted)')
    print(f'  Recall         : {rec:.4f}  (weighted)')
    print(f'  F1 Score       : {f1:.4f}  (weighted)')
    print(f'  MAE            : {mae:.4f}  (rating points)')
    print(f'  Within +/-1    : {within1:.4f}')
    print()
    print(classification_report(y_true, y_pred, zero_division=0))

    model_scores.append({'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec,
                          'F1 Score': f1, 'MAE': mae, 'Within +/-1': within1})

# ## 9. Logistic Regression

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

y_pred_lr = log_reg.predict(X_test_scaled)

evaluate_model('Logistic Regression', y_test, y_pred_lr)

plt.figure(figsize=(6,5))
labels_sorted = sorted(y.unique())
sns.heatmap(confusion_matrix(y_test, y_pred_lr, labels=labels_sorted), annot=True, fmt='d', cmap='Blues',
            xticklabels=labels_sorted, yticklabels=labels_sorted)
plt.title('Logistic Regression - Confusion Matrix')
plt.xlabel('Predicted rating')
plt.ylabel('Actual rating')
plt.show()

# ## 10. Random Forest (with hyperparameter tuning)
# 
# A small `GridSearchCV` (3-fold CV) over `n_estimators` and `max_depth`.

from sklearn.ensemble import RandomForestClassifier

rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_param_grid, cv=3, scoring='f1_weighted', n_jobs=-1
)
rf_grid.fit(X_train, y_train)   # tree-based models don't need scaled features

print('Best params:', rf_grid.best_params_)
rf = rf_grid.best_estimator_

y_pred_rf = rf.predict(X_test)

evaluate_model('Random Forest', y_test, y_pred_rf)

# Feature importance
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)

plt.figure(figsize=(8,6))
importances.plot(kind='barh')
plt.title('Random Forest - Top 15 Feature Importances')
plt.gca().invert_yaxis()
plt.show()

# ## 11. Gradient Descent based classifier
# 
# `SGDClassifier` is a linear classifier trained with (stochastic) gradient descent.

from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
sgd.fit(X_train_scaled, y_train)

y_pred_sgd = sgd.predict(X_test_scaled)

evaluate_model('SGD (Gradient Descent)', y_test, y_pred_sgd)

# #### Bonus: Gradient Descent from scratch (binary, educational)
# 
# A from-scratch batch gradient descent implementation of logistic regression, on a binary version of the problem: **Satisfied** (rating 4-5) vs. **Not satisfied** (rating 1-3).

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_gd(X, y, lr=0.1, epochs=1000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0
    losses = []

    for _ in range(epochs):
        z = X @ weights + bias
        preds = sigmoid(z)

        dw = (1/n_samples) * (X.T @ (preds - y))
        db = (1/n_samples) * np.sum(preds - y)

        weights -= lr * dw
        bias -= lr * db

        loss = -np.mean(y*np.log(preds+1e-9) + (1-y)*np.log(1-preds+1e-9))
        losses.append(loss)

    return weights, bias, losses

# Binary target: Satisfied (4-5) = 1, Not satisfied (1-3) = 0
y_train_bin = (y_train >= 4).astype(int).values
y_test_bin = (y_test >= 4).astype(int).values
majority_baseline = max(y_test_bin.mean(), 1 - y_test_bin.mean())

w, b, losses = train_logistic_gd(X_train_scaled, y_train_bin, lr=0.1, epochs=1000)

plt.plot(losses)
plt.title('Gradient Descent - Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross-Entropy Loss')
plt.show()

preds_bin = (sigmoid(X_test_scaled @ w + b) >= 0.5).astype(int)
bin_acc = accuracy_score(y_test_bin, preds_bin)
print(f'From-scratch Gradient Descent Accuracy (Satisfied vs not): {bin_acc:.4f}')
print(f'Majority-class baseline for this split               : {majority_baseline:.4f}')

# ## 12. Model comparison

results = pd.DataFrame(model_scores).sort_values('F1 Score', ascending=False).reset_index(drop=True)
print(results)

plot_cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
results_melted = results.melt(id_vars='Model', value_vars=plot_cols, var_name='Metric', value_name='Score')

plt.figure(figsize=(9,5))
sns.barplot(data=results_melted, x='Model', y='Score', hue='Metric')
plt.ylim(0,1)
plt.title('Model Comparison - Accuracy, Precision, Recall, F1 (weighted)')
plt.xticks(rotation=10)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ## 13. Robustness check: 5-fold cross-validation
# 
# With only 2,769 rows, a single 80/20 split could be a lucky or unlucky draw. Repeating the evaluation across 5 stratified folds and reporting mean ± standard deviation confirms whether the numbers above are stable, rather than an artifact of one particular split.

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []

for name, model, use_scaled in [
    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42), True),
    ('Random Forest', RandomForestClassifier(n_estimators=rf.n_estimators, max_depth=rf.max_depth, random_state=42, n_jobs=-1), False),
    ('SGD (Gradient Descent)', SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42), True),
]:
    data = X_train_scaled if use_scaled else X_train
    scores = cross_validate(model, data, y_train, cv=cv, scoring=['accuracy', 'f1_weighted'])
    cv_rows.append({
        'Model': name,
        'CV Accuracy (mean)': scores['test_accuracy'].mean(),
        'CV Accuracy (std)': scores['test_accuracy'].std(),
        'CV F1 (mean)': scores['test_f1_weighted'].mean(),
        'CV F1 (std)': scores['test_f1_weighted'].std(),
    })

cv_results = pd.DataFrame(cv_rows)
print(cv_results.round(4))

# ## 14. Conclusion
# 
# **Single 80/20 split:**
# 
# | Model | Accuracy | F1 | MAE | Within ±1 |
# |---|---|---|---|---|
# | SGD (Gradient Descent) | 0.213 | 0.211 | 1.644 | 0.514 |
# | Random Forest (tuned) | 0.193 | 0.190 | 1.551 | 0.540 |
# | Logistic Regression | 0.181 | 0.177 | 1.579 | 0.507 |
# 
# **5-fold cross-validation (on the training set, mean ± std):**
# 
# | Model | CV Accuracy | CV F1 |
# |---|---|---|
# | Logistic Regression | 0.214 ± 0.017 | 0.208 ± 0.016 |
# | Random Forest | 0.202 ± 0.012 | 0.200 ± 0.013 |
# | SGD (Gradient Descent) | 0.198 ± 0.016 | 0.186 ± 0.015 |
# 
# **Three independent checks all point the same way:**
# 
# 1. **The sentiment experiment (Section 6)** confirmed rather than assumed the negative result: correlation between resolution sentiment and satisfaction rating was ~0.01, essentially zero, and only 8% of resolution notes contained any sentiment word at all — because those notes are randomly-generated placeholder sentences, not real agent write-ups.
# 2. **Cross-validation** shows the ~20% accuracy is stable across folds (std of ~0.01–0.02), not a lucky or unlucky single split — every model consistently lands within 1-2 points of the ~20% random-guess baseline for 5 balanced classes.
# 3. **Ordinal-aware metrics** don't rescue the picture either: MAE of ~1.5-1.6 rating points and "within ±1" accuracy of ~51-54% are close to what you'd get by predicting the middle rating for everyone — there's no meaningful ability to even get *close* to the right rating, not just to hit it exactly.
# 
# **Takeaway:** none of the available customer/ticket/product attributes — including a genuinely-tested text feature — carry meaningful predictive signal about `Customer Satisfaction Rating` in this dataset. That's a defensible, evidence-backed conclusion now, not an assumption. The pipeline itself (leakage-aware feature dropping, proper encoding, hyperparameter tuning, cross-validated evaluation, and an honestly-tested NLP feature) is complete and transfers directly to a dataset where the label does carry signal.
