# Churn & NLP Pipeline – Top Findings

## Churn model
- Best model is selected by ROC-AUC among Logistic Regression, Naive Bayes, Decision Tree, KNN, SVM, Random Forest, Gradient Boosting, and Stacking.
- Preprocessing: duplicate rows dropped; numeric missing values imputed with median; categorical with mode; high-cardinality categoricals use frequency encoding; others use one-hot (drop first). Numeric features scaled with StandardScaler.
- Train/test split: 80/20, stratified on churn, random_state=42. Hyperparameter tuning via 5-fold GridSearchCV where specified.
- See `reports/churn_classification_report.txt`, `reports/churn_confusion_matrix.png`, and `reports/roc_curve.png` for evaluation.

## NLP ticket model
- Text = ticket_subject + ticket_description; preprocessing: lowercase, strip HTML/URLs, punctuation; TF-IDF (max_features=5000, ngram_range=(1,2), stop_words='english').
- Classifier: LogisticRegression (or Naive Bayes if &lt;2000 rows). Target: ticket_priority (low/medium/high/critical).
- Macro F1 and per-class metrics in `reports/nlp_classification_report.txt`.

## Decision engine
- Combines churn probability, risk label (High/Medium/Low), and ticket priority into final_status (e.g. Severe churn risk) and summary_signal. High risk + high/critical ticket => Severe; high risk only => High; medium => Medium; low => Low.

## Generative explainer
- If OPENAI_API_KEY is set, uses OpenAI for a short explanation; otherwise deterministic template with risk, probability, ticket priority, top reasons, and one suggested action.
