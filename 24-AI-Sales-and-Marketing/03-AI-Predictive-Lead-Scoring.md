# AI-Powered Predictive Lead Scoring

> **Document Version**: 1.0 — June 2026
> **Scope**: Comprehensive guide on machine learning-based lead scoring, including model architecture, feature engineering, implementation code, evaluation, and CRM integration.

---

## Table of Contents

1. [Introduction to Predictive Lead Scoring](#1-introduction-to-predictive-lead-scoring)
2. [Why ML Beats Rule-Based Scoring](#2-why-ml-beats-rule-based-scoring)
3. [Model Architecture](#3-model-architecture)
4. [Feature Engineering](#4-feature-engineering)
5. [Feature Importance Analysis](#5-feature-importance-analysis)
6. [Model Training Pipeline](#6-model-training-pipeline)
7. [Model Evaluation](#7-model-evaluation)
8. [Real-Time Scoring Architecture](#8-real-time-scoring-architecture)
9. [ABM Account Scoring](#9-abm-account-scoring)
10. [Conversion Prediction](#10-conversion-prediction)
11. [CRM Integration Patterns](#11-crm-integration-patterns)
12. [Model Monitoring and Retraining](#12-model-monitoring-and-retraining)
13. [Implementation Code](#13-implementation-code)
14. [Tools and Vendors](#14-tools-and-vendors)
15. [Best Practices and Pitfalls](#15-best-practices-and-pitfalls)

---

## 1. Introduction to Predictive Lead Scoring

### 1.1 What is Predictive Lead Scoring?

Predictive lead scoring uses machine learning algorithms to automatically rank leads based on their likelihood to convert into customers. Unlike traditional rule-based scoring (where humans define explicit scoring rules like "job title = VP → +20 points"), ML-based scoring learns patterns from historical data to identify which combinations of attributes and behaviors are most predictive of conversion.

### 1.2 Why It Matters

- **Accuracy**: ML models consistently outperform rule-based systems by 25-40% in precision
- **Objectivity**: Eliminates human bias and inconsistent scoring
- **Scalability**: Scores millions of leads in real-time without manual effort
- **Adaptability**: Models automatically adapt to changing market conditions
- **Granularity**: Provides probabilistic scores (0-100%) rather than arbitrary point systems
- **Insights**: Reveals which factors truly drive conversion

### 1.3 Key Metrics Impact

| Metric | Before ML Scoring | After ML Scoring | Improvement |
|--------|-------------------|------------------|-------------|
| Lead-to-Opportunity Conversion | 5-10% | 12-20% | 2x |
| Sales Team Productivity | 30% time on qualified leads | 70% time on qualified leads | 2.3x |
| Marketing Qualified Lead (MQL) Quality | Inconsistent | Highly targeted | 3x better |
| Cost per Lead | Baseline | 30-50% lower | Significant savings |
| Time to Follow-up | Hours to days | Real-time | Dramatically faster |

---

## 2. Why ML Beats Rule-Based Scoring

### 2.1 Limitations of Rule-Based Scoring

```
Rule-Based Scoring:
├── "VP of Sales" title = +25 points
├── "Enterprise" company size = +20 points
├── Downloaded whitepaper = +15 points
├── Visited pricing page = +10 points
└── Total > 80 → MQL

Problems:
❌ Linear thinking — no interaction effects
❌ Arbitrary weights — based on guesses, not data
❌ Static rules — never adapt to market changes
❌ Missing non-linear relationships
❌ Cannot handle high-dimensional data
❌ Reinforces existing biases
```

### 2.2 Advantages of ML-Based Scoring

- **Learns interactions**: Model automatically discovers that "VP of Sales + visited pricing page + intent signal" is much more valuable than the sum of individual signals
- **Data-driven weights**: Weights derived from historical conversion data
- **Adaptive**: Retrains as market conditions change
- **Probabilistic**: Outputs calibrated probability, not arbitrary points
- **Handles complexity**: Works with hundreds of features and their interactions
- **Feature importance**: Identifies which factors actually drive conversions

---

## 3. Model Architecture

### 3.1 Algorithm Comparison

| Algorithm | Pros | Cons | Best For | Typical AUC |
|-----------|------|------|----------|-------------|
| Logistic Regression | Interpretable, fast, low variance | Limited expressiveness, linear | Baseline, when interpretability is critical | 0.70-0.80 |
| Random Forest | Handles non-linearity, feature importance | Can overfit, less interpretable | General purpose, medium datasets | 0.80-0.88 |
| XGBoost/LightGBM | State-of-the-art for tabular data, handles missing values | Requires hyperparameter tuning | Most lead scoring use cases | 0.85-0.93 |
| Neural Network | Captures complex patterns, interactions | Requires large data, less interpretable | Very large datasets, deep feature interactions | 0.85-0.92 |
| Ensemble (Stacking) | Combines strengths of multiple models | More complex to deploy | Maximum performance needed | 0.88-0.95 |

### 3.2 Recommended Architecture

For most B2B lead scoring use cases, we recommend a **gradient boosting ensemble** as the primary model, with **logistic regression** as an interpretable baseline.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ENSEMBLE ARCHITECTURE                           │
│                                                                         │
│  Raw Features                                                           │
│       │                                                                 │
│       ▼                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │
│  │ XGBoost  │  │ LightGBM │  │ CatBoost │  │ Logistic Regression  │  │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘  └──────────┬───────────┘  │
│        │             │             │                  │                │
│        └─────────────┼─────────────┼──────────────────┘                │
│                      │             │                                    │
│                      ▼             ▼                                    │
│              ┌─────────────────────────────┐                           │
│              │    Meta-Learner (Stacking)   │                           │
│              │    Logistic Regression       │                           │
│              │    or Neural Network         │                           │
│              └─────────────┬───────────────┘                           │
│                            │                                            │
│                            ▼                                            │
│                    Final Prediction (0-1)                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Deep Learning for Lead Scoring

For organizations with very large datasets (1M+ leads), deep learning can capture complex feature interactions:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeadScoringNet(nn.Module):
    """
    Deep neural network for lead scoring.
    Handles high-dimensional categorical features via embeddings.
    """
    
    def __init__(self, num_numeric_features, categorical_dims, 
                 embedding_dim=16, hidden_dims=[128, 64, 32]):
        super().__init__()
        
        # Embedding layers for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, embedding_dim)
            for num_classes in categorical_dims
        ])
        
        # Calculate total feature dimension
        cat_embedding_dim = len(categorical_dims) * embedding_dim
        total_input_dim = num_numeric_features + cat_embedding_dim
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        prev_dim = total_input_dim
        for hidden_dim in hidden_dims:
            self.fc_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.fc_layers.append(nn.BatchNorm1d(hidden_dim))
            self.fc_layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        # Output layer
        self.output = nn.Linear(prev_dim, 1)
        
    def forward(self, numeric_features, categorical_features):
        # Process categorical features through embeddings
        embedded = []
        for i, emb in enumerate(self.embeddings):
            embedded.append(emb(categorical_features[:, i]))
        cat_embeddings = torch.cat(embedded, dim=1)
        
        # Concatenate numeric and categorical features
        x = torch.cat([numeric_features, cat_embeddings], dim=1)
        
        # Forward through fully connected layers
        for i in range(0, len(self.fc_layers), 3):
            x = self.fc_layers[i](x)        # Linear
            x = self.fc_layers[i+1](x)      # BatchNorm
            x = F.relu(x)
            x = self.fc_layers[i+2](x)      # Dropout
        
        # Output
        x = self.output(x)
        return torch.sigmoid(x)

# Usage
# model = LeadScoringNet(num_numeric_features=50, categorical_dims=[10, 5, 20, 15])
```

---

## 4. Feature Engineering

### 4.1 Feature Categories

Feature engineering is the most important factor in lead scoring model performance. Well-engineered features can improve AUC by 0.10-0.20 over raw features alone.

#### 4.1.1 Firmographic Features

| Feature | Description | Transformation | Importance |
|---------|-------------|---------------|------------|
| company_size | Number of employees | Log transform | High |
| annual_revenue | Revenue in USD | Log transform | High |
| industry | Industry vertical | One-hot encode | Medium |
| geography | Geographic region | Target encoding | Medium |
| company_age | Years since founding | Log transform | Low-Medium |
| funding_stage | Seed/Series A/B/C/PE | Ordinal encode | High |
| funding_total | Total funding amount | Log transform | Medium |
| ownership_type | Public/Private/Non-profit | One-hot encode | Low |
| growth_rate | YoY employee growth | Raw value | High |
| tech_stack_score | % overlap with ICP tech | Min-max scale | High |

#### 4.1.2 Behavioral Features

| Feature | Description | Window | Importance |
|---------|-------------|--------|------------|
| page_views_total | Total website page views | 90 days | Medium |
| page_views_pricing | Pricing page views | 30 days | Very High |
| page_views_features | Features page views | 30 days | High |
| content_downloads | Whitepapers, ebooks downloaded | 90 days | High |
| webinar_attendance | Webinars attended (count) | 180 days | Medium |
| email_opens | Marketing email opens | 90 days | Medium |
| email_clicks | Marketing email clicks | 90 days | Medium |
| form_fills | Forms completed | 90 days | High |
| trial_started | Product trial initiated | Binary | Very High |
| trial_actions | Actions taken in trial | 30 days | Very High |
| support_tickets | Support tickets created | 180 days | Low |
| event_attendance | In-person event attendance | 365 days | Medium |
| social_engagement | LinkedIn/Twitter engagement | 90 days | Low |
| referral_count | Referral visits | 90 days | Medium |

#### 4.1.3 Intent Features

| Feature | Description | Source | Importance |
|---------|-------------|--------|------------|
| intent_score | Composite intent score | 6sense, Bombora | Very High |
| topic_clusters | Topic categories showing intent | Intent providers | High |
| search_volume | Search volume for target keywords | SEMrush, Ahrefs | Medium |
| competitor_research | Researching competitors | Intent providers | High |
| job_postings | Hiring in target departments | LinkedIn, Indeed | Medium |
| technology_change | Tech stack changes | BuiltWith, Datanyze | Medium |
| funding_events | Recent funding rounds | Crunchbase | Medium |
| leadership_changes | C-suite changes | LinkedIn | Low-Medium |
| regulatory_changes | Industry regulation changes | News sources | Low |
| seasonality | Buying season patterns | Derived | Medium |

#### 4.1.4 Derived Features

```python
def engineer_lead_features(leads_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer advanced features for lead scoring model.
    """
    df = leads_df.copy()
    
    # Recency features
    current_date = pd.Timestamp.now()
    df['days_since_first_touch'] = (
        current_date - pd.to_datetime(df['first_touch_date'])
    ).dt.days
    df['days_since_last_activity'] = (
        current_date - pd.to_datetime(df['last_activity_date'])
    ).dt.days
    df['days_since_last_email_open'] = (
        current_date - pd.to_datetime(df['last_email_open_date'])
    ).dt.days
    df['days_since_last_visit'] = (
        current_date - pd.to_datetime(df['last_visit_date'])
    ).dt.days
    
    # Engagement velocity
    df['engagement_per_day'] = (
        df['total_engagements'] / (df['days_since_first_touch'] + 1)
    )
    df['recent_engagement_velocity'] = (
        df['engagements_last_30_days'] / 30
    )
    df['engagement_acceleration'] = (
        df['engagements_last_30_days'] / 
        (df['engagements_last_90_days'] - df['engagements_last_30_days'] + 1)
    )
    
    # Content affinity
    content_columns = [c for c in df.columns if c.startswith('content_')]
    df['content_diversity'] = (df[content_columns] > 0).sum(axis=1)
    df['high_value_content_engagement'] = (
        df[['content_case_studies', 'content_pricing', 
            'content_whitepapers', 'content_demo']].sum(axis=1)
    )
    
    # Channel efficiency
    df['channel_concentration'] = (
        df[['email_opens', 'website_visits', 'social_clicks']].max(axis=1) /
        (df[['email_opens', 'website_visits', 'social_clicks']].sum(axis=1) + 1)
    )
    df['multi_channel_score'] = (
        (df['email_opens'] > 0).astype(int) +
        (df['website_visits'] > 0).astype(int) +
        (df['social_clicks'] > 0).astype(int) +
        (df['event_attendance'] > 0).astype(int)
    )
    
    # Interaction features
    df['pricing_x_intent'] = df['page_views_pricing'] * df['intent_score']
    df['trial_x_fit'] = df['trial_started'] * df['firmographic_fit_score']
    df['engagement_x_recency'] = (
        df['engagement_score'] * (1 / (df['days_since_last_activity'] + 1))
    )
    
    # Time-based features
    df['is_quarter_end'] = (pd.to_datetime(df['created_date']).dt.month % 3 == 0).astype(int)
    df['is_year_end'] = (pd.to_datetime(df['created_date']).dt.month == 12).astype(int)
    df['created_month'] = pd.to_datetime(df['created_date']).dt.month
    df['created_quarter'] = pd.to_datetime(df['created_date']).dt.quarter
    
    # Gap features
    df['gap_between_touch_and_response'] = (
        (pd.to_datetime(df['first_response_date']) - 
         pd.to_datetime(df['first_touch_date'])).dt.days
    )
    
    # Lead source quality score (historical conversion rate by source)
    source_conv_rates = df.groupby('lead_source')['converted'].mean().to_dict()
    df['source_quality_score'] = df['lead_source'].map(source_conv_rates)
    
    # Company peer conversion rate
    industry_conv_rates = df.groupby('industry')['converted'].mean().to_dict()
    df['industry_conversion_rate'] = df['industry'].map(industry_conv_rates)
    
    return df
```

### 4.2 Feature Selection

```python
def select_features(X, y, n_features=30):
    """Select most important features using multiple methods."""
    from sklearn.feature_selection import (
        SelectFromModel, SelectKBest, f_classif, mutual_info_classif
    )
    import xgboost as xgb
    
    results = {}
    
    # Method 1: XGBoost importance
    model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    xgb_importances = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)
    results['xgb_top'] = xgb_importances.head(n_features).index.tolist()
    
    # Method 2: Mutual information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    results['mi_top'] = mi_series.head(n_features).index.tolist()
    
    # Method 3: Correlation with target
    correlations = X.apply(lambda col: col.corr(pd.Series(y)))
    corr_series = correlations.abs().sort_values(ascending=False)
    results['corr_top'] = corr_series.head(n_features).index.tolist()
    
    # Consensus features (appearing in top N of at least 2 methods)
    from collections import Counter
    feature_votes = Counter()
    for method, features in results.items():
        for feat in features:
            feature_votes[feat] += 1
    
    consensus = [feat for feat, votes in feature_votes.most_common() if votes >= 2]
    
    return {
        'all_features': list(X.columns),
        'selected_features': consensus,
        'feature_rankings': results,
        'xgb_importances': xgb_importances.to_dict(),
        'n_selected': len(consensus)
    }
```

---

## 5. Feature Importance Analysis

### 5.1 Interpreting Feature Importance

Understanding which features drive lead scoring decisions provides business insights and builds trust in the model.

```python
def analyze_feature_importance(model, feature_names, top_n=20):
    """
    Comprehensive feature importance analysis.
    Uses multiple importance measures for robust understanding.
    """
    import shap
    
    results = {}
    
    # Method 1: Built-in importance (gain/weight)
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance_gain': model.feature_importances_
        }).sort_values('importance_gain', ascending=False)
        results['gain_importance'] = importance
    
    # Method 2: SHAP values (most robust)
    explainer = shap.TreeExplainer(model)
    # Use a sample for efficiency
    sample_data = X_test.sample(min(1000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(sample_data)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # For multi-class
    
    shap_summary = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0),
        'shap_std': np.std(shap_values, axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    
    results['shap_importance'] = shap_summary
    
    # Method 3: Permutation importance
    perm_importance = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42
    )
    
    perm_df = pd.DataFrame({
        'feature': feature_names,
        'permutation_importance': perm_importance.importances_mean,
        'permutation_std': perm_importance.importances_std
    }).sort_values('permutation_importance', ascending=False)
    
    results['permutation_importance'] = perm_df
    
    # Method 4: Partial dependence (for top features)
    from sklearn.inspection import partial_dependence
    
    top_features = results['shap_importance']['feature'].head(5).tolist()
    pd_results = partial_dependence(
        model, X_test, top_features, kind='average'
    )
    results['partial_dependence'] = pd_results
    
    return results

# Visualize SHAP summary
def plot_shap_summary(shap_values, X_sample, feature_names, max_display=20):
    """Create SHAP summary plot."""
    shap.summary_plot(
        shap_values, X_sample, 
        feature_names=feature_names,
        max_display=max_display,
        show=False
    )
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
```

### 5.2 Typical Feature Importance Rankings

Based on analysis across 50+ B2B lead scoring models, here are typical importance rankings:

| Rank | Feature Category | Specific Feature | Avg. Importance | Business Insight |
|------|-----------------|-----------------|----------------|-----------------|
| 1 | Behavioral | trial_started | 0.18 | Product experience is strongest signal |
| 2 | Intent | intent_score | 0.14 | Being in-market is highly predictive |
| 3 | Behavioral | pricing_page_views_30d | 0.11 | Pricing research indicates purchase intent |
| 4 | Firmographic | company_size_fit | 0.09 | ICP fit still matters |
| 5 | Behavioral | engagement_acceleration | 0.08 | Increasing engagement predicts conversion |
| 6 | Behavioral | content_downloads_high_value | 0.07 | Deep content engagement > superficial |
| 7 | Firmographic | tech_stack_overlap | 0.06 | Tech compatibility matters |
| 8 | Behavioral | days_since_last_activity | 0.05 | Recency is a strong signal |
| 9 | Derived | pricing_x_intent_interaction | 0.05 | Combined signals > individual |
| 10 | Firmographic | industry | 0.04 | Some industries convert better |

### 5.3 SHAP Analysis Example Output

```
Feature Importance by Mean |SHAP| Value:

intent_score                  ████████████████████ 0.142
trial_started                 ████████████████████ 0.138
pricing_page_views_30d       ████████████████     0.112
company_size                  ████████████         0.082
engagement_acceleration       ████████████         0.078
content_downloads             █████████            0.065
tech_stack_overlap            ██████               0.058
days_since_last_activity      ██████               0.052
pricing_x_intent              █████                0.048
industry                      ████                 0.036
email_click_rate              ███                  0.028
webinar_attendance            ███                  0.025
referral_count                ██                   0.018
days_since_first_touch        ██                   0.015
channel_diversity              █                   0.010
```

---

## 6. Model Training Pipeline

### 6.1 End-to-End Pipeline

```python
"""
Complete lead scoring model training pipeline.
Includes data preparation, feature engineering, model training,
hyperparameter optimization, and evaluation.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    average_precision_score, log_loss
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
import optuna
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class LeadScoringPipeline:
    """
    Full-stack lead scoring pipeline with hyperparameter optimization.
    """
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.training_metadata = {}
        
    def _load_config(self, path: str = None) -> dict:
        """Default configuration."""
        return {
            'model_type': 'xgboost',
            'test_size': 0.2,
            'validation_size': 0.1,
            'random_state': 42,
            'cv_folds': 5,
            'optimize_hyperparams': True,
            'n_optuna_trials': 100,
            'scale_pos_weight': 'auto',  # Handle class imbalance
            'early_stopping_rounds': 50,
            'n_estimators': 1000,
            'feature_selection': True,
            'n_features_to_select': 30
        }
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'converted',
                     exclude_cols: list = None):
        """
        Prepare data for model training.
        
        Args:
            df: Raw lead data with features and target
            target_col: Name of target column (binary)
            exclude_cols: Columns to exclude (IDs, dates, text)
        """
        if exclude_cols is None:
            exclude_cols = ['lead_id', 'contact_id', 'account_id', 
                          'name', 'email', 'phone', 'created_date']
        
        # Separate target
        y = df[target_col].values
        
        # Remove excluded columns
        feature_df = df.drop(columns=[target_col] + exclude_cols, errors='ignore')
        
        # Identify feature types
        self.numeric_cols = feature_df.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        
        self.categorical_cols = feature_df.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        self.feature_names = feature_df.columns.tolist()
        
        # Handle missing values and encode
        X = self._preprocess_features(feature_df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state'],
            stratify=y
        )
        
        # Further split training for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, 
            test_size=self.config['validation_size'],
            random_state=self.config['random_state'],
            stratify=y_train
        )
        
        # Class weights for imbalanced data
        if self.config['scale_pos_weight'] == 'auto':
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        else:
            scale_pos_weight = self.config['scale_pos_weight']
        
        self.training_metadata = {
            'n_samples': len(df),
            'n_features': len(self.feature_names),
            'n_positive': int(y.sum()),
            'n_negative': int((1 - y).sum()),
            'positive_rate': float(y.mean()),
            'scale_pos_weight': float(scale_pos_weight),
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test),
            'numeric_features': len(self.numeric_cols),
            'categorical_features': len(self.categorical_cols)
        }
        
        return X_train, X_val, X_test, y_train, y_val, y_test, scale_pos_weight
    
    def _preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features with appropriate transformations."""
        df = df.copy()
        
        # Handle numeric features
        for col in self.numeric_cols:
            # Fill missing with median
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
            # Clip outliers
            q1 = df[col].quantile(0.01)
            q3 = df[col].quantile(0.99)
            df[col] = df[col].clip(q1, q3)
        
        # Handle categorical features
        for col in self.categorical_cols:
            # Fill missing with 'Unknown'
            df[col] = df[col].fillna('Unknown')
            # Encode to integer
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        
        return df
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val,
                                  scale_pos_weight: float) -> dict:
        """Bayesian hyperparameter optimization with Optuna."""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'scale_pos_weight': scale_pos_weight,
                'eval_metric': 'auc',
                'random_state': self.config['random_state'],
                'n_jobs': -1
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=self.config['early_stopping_rounds'],
                verbose=False
            )
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            return auc
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.config['random_state']),
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(objective, n_trials=self.config['n_optuna_trials'])
        
        print(f"Best trial: AUC {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
        
        return study.best_params
    
    def train(self, X_train, y_train, X_val, y_val, params: dict = None):
        """Train the lead scoring model."""
        
        if params is None:
            # Use default parameters
            params = {
                'n_estimators': self.config['n_estimators'],
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': self.training_metadata['scale_pos_weight'],
                'eval_metric': 'auc',
                'random_state': self.config['random_state'],
                'n_jobs': -1
            }
        
        self.model = xgb.XGBClassifier(**params)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=self.config['early_stopping_rounds'],
            verbose=100
        )
        
        self.training_metadata['best_iteration'] = self.model.best_iteration
        self.training_metadata['training_completed'] = datetime.now().isoformat()
        
        return self.model
    
    def evaluate(self, X_test, y_test) -> dict:
        """Comprehensive model evaluation."""
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        auc_pr = average_precision_score(y_test, y_pred_proba)
        
        # Precision/Recall at different thresholds
        metrics = {
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'log_loss': log_loss(y_test, y_pred_proba),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Lift analysis
        metrics['lift_analysis'] = self._calculate_lift(y_test, y_pred_proba)
        
        # Threshold analysis for optimal cutoffs
        metrics['threshold_analysis'] = self._analyze_thresholds(y_test, y_pred_proba)
        
        return metrics
    
    def _calculate_lift(self, y_test, y_pred_proba, deciles=10) -> dict:
        """Calculate lift at each decile."""
        df = pd.DataFrame({'actual': y_test, 'predicted': y_pred_proba})
        df['decile'] = pd.qcut(df['predicted'].rank(method='first'), 
                               q=deciles, labels=False, duplicates='drop')
        df['decile'] = deciles - df['decile']  # Reverse so decile 1 = highest scores
        
        lift_data = []
        baseline_rate = y_test.mean()
        
        for d in range(1, deciles + 1):
            decile_data = df[df['decile'] == d]
            if len(decile_data) == 0:
                continue
            actual_rate = decile_data['actual'].mean()
            lift = actual_rate / baseline_rate if baseline_rate > 0 else 0
            cum_conversions = df[df['decile'] <= d]['actual'].sum()
            total_conversions = y_test.sum()
            cum_conversions_pct = cum_conversions / total_conversions if total_conversions > 0 else 0
            
            lift_data.append({
                'decile': d,
                'n_leads': len(decile_data),
                'actual_rate': actual_rate,
                'lift': lift,
                'cumulative_conversions_pct': cum_conversions_pct
            })
        
        return {
            'baseline_rate': baseline_rate,
            'deciles': lift_data,
            'lift_at_10pct': lift_data[0]['lift'] if lift_data else 0,
            'cumulative_at_50pct': lift_data[4]['cumulative_conversions_pct'] if len(lift_data) > 4 else 0
        }
    
    def _analyze_thresholds(self, y_test, y_pred_proba) -> list:
        """Analyze precision/recall at different thresholds."""
        thresholds = np.arange(0.05, 0.95, 0.05)
        results = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            
            precision = precision_score(y_test, y_pred_thresh, zero_division=0)
            recall = recall_score(y_test, y_pred_thresh, zero_division=0)
            f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'n_predicted_positive': int(y_pred_thresh.sum())
            })
        
        return results
    
    def save_model(self, path: str):
        """Save model and metadata."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        model_path = f"{path}/model.json"
        self.model.save_model(model_path)
        
        # Save metadata
        metadata_path = f"{path}/metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.training_metadata, f, indent=2, default=str)
        
        # Save feature names
        features_path = f"{path}/features.json"
        with open(features_path, 'w') as f:
            json.dump({
                'all_features': self.feature_names,
                'numeric_features': self.numeric_cols,
                'categorical_features': self.categorical_cols
            }, f, indent=2)
        
        print(f"Model saved to {model_path}")
        print(f"Metadata saved to {metadata_path}")
    
    def load_model(self, path: str):
        """Load saved model and metadata."""
        model_path = f"{path}/model.json"
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        
        with open(f"{path}/metadata.json", 'r') as f:
            self.training_metadata = json.load(f)
        
        with open(f"{path}/features.json", 'r') as f:
            features = json.load(f)
            self.feature_names = features['all_features']
            self.numeric_cols = features['numeric_features']
            self.categorical_cols = features['categorical_features']
        
        print(f"Model loaded from {model_path}")
        return self
    
    def predict(self, leads_df: pd.DataFrame) -> np.ndarray:
        """Generate predictions for new leads."""
        X = self._preprocess_features(leads_df[self.feature_names])
        return self.model.predict_proba(X)[:, 1]

# ===== Example Usage =====

# Load data
leads = pd.read_csv('leads_historical.csv')
leads['converted'] = (leads['opportunity_created'].notna()).astype(int)

# Initialize pipeline
pipeline = LeadScoringPipeline()
pipeline.config['optimize_hyperparams'] = True
pipeline.config['n_optuna_trials'] = 50

# Prepare data
X_train, X_val, X_test, y_train, y_val, y_test, sw = pipeline.prepare_data(
    leads, target_col='converted'
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")  
print(f"Test samples: {len(X_test)}")
print(f"Positive rate: {pipeline.training_metadata['positive_rate']:.2%}")
print(f"Scale pos weight: {sw:.2f}")

# Optimize hyperparameters
if pipeline.config['optimize_hyperparams']:
    best_params = pipeline.optimize_hyperparameters(
        X_train, y_train, X_val, y_val, sw
    )
else:
    best_params = None

# Train model
pipeline.train(X_train, y_train, X_val, y_val, best_params)

# Evaluate
metrics = pipeline.evaluate(X_test, y_test)
print(f"\nAUC-ROC: {metrics['auc_roc']:.4f}")
print(f"AUC-PR: {metrics['auc_pr']:.4f}")
print(f"Lift at 10%: {metrics['lift_analysis']['lift_at_10pct']:.2f}x")

# Save
pipeline.save_model('models/lead_scoring_v1')
```

### 6.2 Handling Class Imbalance

Lead scoring datasets are typically highly imbalanced (1-10% conversion rate). Key strategies:

1. **Scale Pos Weight**: Set `scale_pos_weight = neg_count / pos_count` in XGBoost
2. **SMOTE**: Synthetic Minority Over-sampling Technique
3. **Stratified Sampling**: Maintain class proportions in train/test splits
4. **Threshold Tuning**: Adjust decision threshold based on business needs
5. **Cost-Sensitive Learning**: Assign different misclassification costs

```python
def handle_class_imbalance(X_train, y_train, method='smote'):
    """Handle class imbalance in training data."""
    
    if method == 'smote':
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return X_resampled, y_resampled
    
    elif method == 'random_oversample':
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=42)
        return ros.fit_resample(X_train, y_train)
    
    elif method == 'adasyn':
        from imblearn.over_sampling import ADASYN
        ada = ADASYN(random_state=42)
        return ada.fit_resample(X_train, y_train)
    
    return X_train, y_train
```

---

## 7. Model Evaluation

### 7.1 Evaluation Metrics

| Metric | Formula | What it Measures | Target |
|--------|---------|-----------------|--------|
| AUC-ROC | Area under ROC curve | Rank-ordering ability (true positive rate vs. false positive rate) | >0.85 |
| AUC-PR | Area under Precision-Recall curve | Performance on positive class | >0.50 |
| Lift at 10% | Conversion rate in top 10% / baseline rate | How much better than random | >5x |
| Precision at K | TP / (TP + FP) in top K leads | Accuracy of lead prioritization | >0.60 |
| Recall at K | TP / (TP + FN) in top K leads | Coverage of high-quality leads | >0.40 |
| F1 Score | 2 * (P * R) / (P + R) | Balanced precision-recall | >0.40 |
| Log Loss | -Σ(y*log(p) + (1-y)*log(1-p)) | Model calibration | <0.30 |

### 7.2 Lift Chart Interpretation

```
Lift Chart — Lead Scoring Model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Decile  |  Leads  |  Conversions  |  Rate   |  Lift  |  Cum. %
────────┼─────────┼──────────────┼─────────┼────────┼─────────
 1      │  1,000  │    185       │  18.5%  │  6.2x  │  38.5%
 2      │  1,000  │     95       │   9.5%  │  3.2x  │  58.3%
 3      │  1,000  │     58       │   5.8%  │  1.9x  │  70.4%
 4      │  1,000  │     38       │   3.8%  │  1.3x  │  78.3%
 5      │  1,000  │     28       │   2.8%  │  0.9x  │  84.2%
 6      │  1,000  │     22       │   2.2%  │  0.7x  │  88.8%
 7      │  1,000  │     18       │   1.8%  │  0.6x  │  92.6%
 8      │  1,000  │     12       │   1.2%  │  0.4x  │  95.1%
 9      │  1,000  │      8       │   0.8%  │  0.3x  │  96.8%
10      │  1,000  │      6       │   0.6%  │  0.2x  │ 100.0%
────────┼─────────┼──────────────┼─────────┼────────┼─────────
Total   │ 10,000  │    470       │   4.7%  │  1.0x  │ 100.0%

Interpretation:
- Top decile: 6.2x better than random (18.5% vs. 4.7%)
- Top 2 deciles: capture 58.3% of all conversions
- Deciles 6-10: below baseline — deprioritize these leads
```

### 7.3 Probability Calibration

Well-calibrated probabilities are crucial for lead scoring. A score of 0.30 should mean that 30% of leads with that score actually convert.

```python
def calibrate_probabilities(model, X_cal, y_cal, method='isotonic'):
    """
    Calibrate predicted probabilities to be well-calibrated.
    """
    from sklearn.calibration import CalibratedClassifierCV
    
    calibrated = CalibratedClassifierCV(
        model, method=method, cv='prefit'
    )
    calibrated.fit(X_cal, y_cal)
    return calibrated

def plot_calibration_curve(y_true, y_pred_proba, n_bins=10):
    """Plot reliability diagram."""
    from sklearn.calibration import calibration_curve
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, 'bo-', 
             label='Model')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('calibration_curve.png', dpi=150)
    plt.close()
```

---

## 8. Real-Time Scoring Architecture

### 8.1 Architecture Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVENT SOURCES                                │
│  Website Actions  │  Email Sends  │  CRM Updates  │  Integrations│
└───────────────────┴───────────────┴───────────────┴─────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVENT STREAM (Kafka/Kinesis)                  │
│                     •  Web visits                               │
│                     •  Email opens/clicks                       │
│                     •  Form submissions                         │
│                     •  Content downloads                        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    REAL-TIME FEATURE COMPUTATION                 │
│                     •  Update counters (page views, opens)      │
│                     •  Calculate time-based features            │
│                     •  Compute engagement velocity              │
│                     •  Update recency features                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE STORE                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Online Store │  │ Offline Store│  │ Feature Definitions    │  │
│  │ (Redis)      │  │ (S3/Snowflake)│  │ (Feast/Tecton)        │  │
│  └──────┬──────┘  └──────┬──────┘  └─────────────────────────┘  │
└─────────┼─────────────────┼─────────────────────────────────────┘
          │                 │
          ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL SERVING                                 │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐  │
│  │ Model Inference │  │ Batch Scoring  │  │ A/B Model Compare│  │
│  │ (<100ms)        │  │ (Hourly/Daily)  │  │                  │  │
│  └───────┬────────┘  └───────┬────────┘  └──────────────────┘  │
└──────────┼────────────────────┼─────────────────────────────────┘
           │                    │
           ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SCORE CONSUMERS                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ CRM Update    │  │ Lead Routing  │  │ Marketing Actions   │  │
│  │ (SFDC/HubSpot)│  │ (Assign SDR)  │  │ (Send to nurture)  │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Real-Time Scoring Endpoint

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import redis
import json
from typing import Optional, List, Dict

app = FastAPI(title="Lead Scoring API")

# Connect to Redis feature store
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Load model
model = xgb.XGBClassifier()
model.load_model('models/lead_scoring_v1/model.json')

# Load feature metadata
with open('models/lead_scoring_v1/features.json', 'r') as f:
    FEATURES = json.load(f)

class LeadEvent(BaseModel):
    lead_id: str
    event_type: str  # 'page_view', 'email_click', 'form_fill', etc.
    event_data: Dict
    timestamp: Optional[str] = None

class ScoreResponse(BaseModel):
    lead_id: str
    score: float
    score_tier: str
    features_used: int
    response_time_ms: float

@app.post("/score/", response_model=ScoreResponse)
async def score_lead(event: LeadEvent):
    """Real-time lead scoring endpoint."""
    import time
    start = time.time()
    
    # Get existing lead features from feature store
    lead_features_key = f"lead:{event.lead_id}:features"
    cached_features = redis_client.get(lead_features_key)
    
    if cached_features:
        features = json.loads(cached_features)
    else:
        # Initialize new lead features
        features = {f: 0 for f in FEATURES['all_features']}
    
    # Update features based on event
    features = update_features(features, event)
    
    # Cache updated features
    redis_client.setex(lead_features_key, 86400, json.dumps(features))  # 24h TTL
    
    # Prepare feature vector
    X = prepare_feature_vector(features)
    
    # Score
    score = float(model.predict_proba([X])[0, 1])
    
    # Determine tier
    if score >= 0.7:
        tier = "hot"
    elif score >= 0.3:
        tier = "warm"
    elif score >= 0.1:
        tier = "cool"
    else:
        tier = "cold"
    
    response_time = (time.time() - start) * 1000
    
    return ScoreResponse(
        lead_id=event.lead_id,
        score=round(score, 4),
        score_tier=tier,
        features_used=len(FEATURES['all_features']),
        response_time_ms=round(response_time, 2)
    )

def update_features(features: Dict, event: LeadEvent) -> Dict:
    """Update feature values based on event."""
    event_type = event.event_type
    data = event.event_data
    
    if event_type == 'page_view':
        page_type = data.get('page_type', '')
        features['website_visits'] = features.get('website_visits', 0) + 1
        if page_type == 'pricing':
            features['pricing_page_views'] = features.get('pricing_page_views', 0) + 1
        elif page_type == 'features':
            features['features_page_views'] = features.get('features_page_views', 0) + 1
    
    elif event_type == 'email_click':
        features['email_clicks'] = features.get('email_clicks', 0) + 1
    
    elif event_type == 'form_fill':
        form_type = data.get('form_type', '')
        features['form_submissions'] = features.get('form_submissions', 0) + 1
        if form_type == 'demo_request':
            features['demo_requested'] = 1
    
    elif event_type == 'content_download':
        features['content_downloads'] = features.get('content_downloads', 0) + 1
    
    # Update recency
    from datetime import datetime
    features['days_since_last_activity'] = 0
    
    return features

def prepare_feature_vector(features: Dict) -> List:
    """Convert features dict to model input vector."""
    X = []
    for feature in FEATURES['all_features']:
        X.append(features.get(feature, 0))
    return X
```

---

## 9. ABM Account Scoring

### 9.1 Account-Level Scoring

For Account-Based Marketing (ABM), we need to score accounts, not just individual leads:

```python
class AccountScoringEngine:
    """
    Account-Based Marketing (ABM) scoring.
    Aggregates individual lead scores into account-level scores
    with additional account-level features.
    """
    
    def __init__(self, lead_scoring_model):
        self.lead_scorer = lead_scoring_model
        
    def score_account(self, account_id: str, account_data: dict,
                      leads: pd.DataFrame) -> dict:
        """
        Score an account based on aggregate lead information
        and account-level attributes.
        """
        # Score each lead in the account
        lead_scores = self.lead_scorer.predict(leads)
        
        # Aggregate lead scores
        top_leads = sorted(lead_scores, reverse=True)[:5]
        
        account_score = self._calculate_account_score(
            account_data=account_data,
            lead_scores=lead_scores,
            n_leads=len(leads),
            n_engaged_leads=(pd.Series(lead_scores) > 0.3).sum(),
            n_hot_leads=(pd.Series(lead_scores) > 0.7).sum()
        )
        
        return {
            'account_id': account_id,
            'account_score': account_score,
            'account_tier': self._get_tier(account_score),
            'n_contacts': len(leads),
            'n_engaged_contacts': int((pd.Series(lead_scores) > 0.3).sum()),
            'n_hot_contacts': int((pd.Series(lead_scores) > 0.7).sum()),
            'max_lead_score': float(max(lead_scores)) if len(lead_scores) > 0 else 0,
            'avg_lead_score': float(np.mean(lead_scores)) if len(lead_scores) > 0 else 0,
            'buying_committee_coverage': self._calculate_coverage(leads),
            'recommended_action': self._recommend_action(account_score, account_data)
        }
    
    def _calculate_account_score(self, account_data: dict, lead_scores: list,
                                   n_leads: int, n_engaged_leads: int,
                                   n_hot_leads: int) -> float:
        """
        Weighted account score combining:
        - Aggregate lead signals (60%)
        - Account fit signals (40%)
        """
        # Lead signal component (60%)
        if n_leads > 0:
            max_score = max(lead_scores) * 0.3
            mean_score = np.mean(lead_scores) * 0.15
            depth_score = (n_engaged_leads / max(n_leads, 1)) * 0.1
            hot_density = (n_hot_leads / max(n_leads, 1)) * 0.05
            lead_component = max_score + mean_score + depth_score + hot_density
        else:
            lead_component = 0
        
        # Account fit component (40%)
        fit_score = account_data.get('firmographic_fit', 0) * 0.15
        intent_score = account_data.get('intent_score', 0) * 0.10
        tech_score = account_data.get('tech_fit', 0) * 0.08
        relationship_score = account_data.get('existing_relationship', 0) * 0.07
        account_component = fit_score + intent_score + tech_score + relationship_score
        
        return min(lead_component + account_component, 1.0)
```

### 9.2 Buying Committee Scoring

Complex B2B sales involve multiple stakeholders. Account scoring should consider:

- **Coverage**: Are we reaching all key roles (economic buyer, technical evaluator, champion)?
- **Consensus**: Are multiple stakeholders engaged positively?
- **Depth**: How deep is engagement within each role?
- **Speed**: Is the account moving faster or slower than typical?

---

## 10. Conversion Prediction

### 10.1 Multi-Stage Conversion Prediction

```python
class MultiStageConversionPredictor:
    """
    Predicts conversion at each stage of the funnel.
    Enables stage-specific routing and treatment.
    """
    
    def __init__(self):
        self.stage_models = {
            'lead_to_mql': self._train_stage_model('lead_to_mql'),
            'mql_to_sql': self._train_stage_model('mql_to_sql'),
            'sql_to_opportunity': self._train_stage_model('sql_to_opportunity'),
            'opportunity_to_close': self._train_stage_model('opportunity_to_close')
        }
    
    def predict_funnel_conversion(self, lead: dict) -> dict:
        """Predict probability at each funnel stage."""
        predictions = {}
        for stage, model in self.stage_models.items():
            predictions[stage] = model.predict_proba(
                self._extract_stage_features(lead, stage)
            )[0, 1]
        
        # Overall predicted probability
        overall = 1.0
        for stage_prob in predictions.values():
            overall *= stage_prob
        
        predictions['overall_conversion_probability'] = overall
        
        # Identify biggest drop-off risk
        predictions['biggest_risk_stage'] = min(
            predictions, key=predictions.get
        )
        
        return predictions
    
    def _extract_stage_features(self, lead: dict, stage: str) -> pd.DataFrame:
        """Extract features relevant to specific stage."""
        # Different stages have different predictive features
        stage_features = {
            'lead_to_mql': ['source', 'company_size', 'industry', 'intent_score'],
            'mql_to_sql': ['email_engagement', 'content_downloads', 'webinar_attendance'],
            'sql_to_opportunity': ['demo_requested', 'trial_actions', 'call_engagement'],
            'opportunity_to_close': ['deal_size', 'competition', 'decision_timeline', 'champion']
        }
        
        relevant_features = stage_features.get(stage, [])
        return pd.DataFrame([{f: lead.get(f, 0) for f in relevant_features}])
```

---

## 11. CRM Integration Patterns

### 11.1 Salesforce Integration

```python
class SalesforceScoringIntegration:
    """
    Bidirectional integration between lead scoring model and Salesforce.
    """
    
    def __init__(self, sf_client, model):
        self.sf = sf_client
        self.model = model
        
    def update_lead_scores(self, lead_ids: list = None):
        """Batch update lead scores in Salesforce."""
        
        if lead_ids:
            leads = self.sf.query(
                f"SELECT Id, {self._get_fields()} FROM Lead WHERE Id IN {tuple(lead_ids)}"
            )
        else:
            leads = self.sf.query(
                f"SELECT Id, {self._get_fields()} FROM Lead WHERE AI_Score__c = null"
            )
        
        for lead_batch in self._batch(leads, size=200):
            features = self._extract_features(lead_batch)
            scores = self.model.predict(features)
            
            for lead, score in zip(lead_batch, scores):
                self.sf.update('Lead', lead['Id'], {
                    'AI_Score__c': float(score),
                    'AI_Score_Tier__c': self._get_tier(score),
                    'AI_Score_Date__c': datetime.now().isoformat(),
                    'AI_Score_Model_Version__c': self.model.version
                })
    
    def subscribe_to_lead_events(self):
        """Subscribe to Salesforce Platform Events for real-time scoring."""
        from simple_salesforce.streaming import PushTopic
        
        push_topic = PushTopic(
            self.sf, 'LeadActivityUpdates',
            "SELECT Id, Email_Opened__c, Website_Visited__c, Form_Filled__c "
            "FROM LeadActivityEvent__e"
        )
        
        push_topic.subscribe(self._handle_lead_event)
    
    def _handle_lead_event(self, event):
        """Handle real-time lead engagement event."""
        lead_id = event.payload['Id']
        
        # Get full lead data
        lead = self.sf.get('Lead', lead_id)
        
        # Score
        features = self._extract_features([lead])
        score = self.model.predict(features)[0]
        
        # Update
        self.sf.update('Lead', lead_id, {
            'AI_Score__c': float(score),
            'Last_Scored__c': datetime.now().isoformat()
        })
        
        # Trigger workflow if high score
        if score > 0.8:
            self._notify_sdr(lead, score)
```

### 11.2 HubSpot Integration

```python
class HubSpotScoringIntegration:
    """
    Integrate lead scoring with HubSpot CRM.
    """
    
    def __init__(self, hubspot_client, model):
        self.hs = hubspot_client
        self.model = model
    
    def update_contact_scores(self):
        """Update contact scores via HubSpot API."""
        from hubspot.crm.contacts import ApiException
        
        # Get contacts with property groups
        contacts = self.hs.crm.contacts.get_all(
            properties=['email', 'hs_lead_status', 'createdate'],
            limit=100
        )
        
        for contact in contacts:
            features = self._extract_contact_features(contact)
            score = self.model.predict([features])[0]
            
            try:
                self.hs.crm.contacts.basic_api.update(
                    contact_id=contact.id,
                    contact_update={
                        'properties': {
                            'ai_lead_score': str(round(score, 2)),
                            'ai_score_tier': self._get_tier(score)
                        }
                    }
                )
            except ApiException as e:
                print(f"Error updating contact {contact.id}: {e}")
```

### 11.3 Custom CRM Integration

```python
class CustomCRMIntegration:
    """
    Generic REST API integration for custom CRM systems.
    """
    
    def __init__(self, api_endpoint: str, api_key: str, model):
        self.endpoint = api_endpoint
        self.api_key = api_key
        self.model = model
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    def push_scores(self, leads: List[Dict]):
        """Push lead scores to CRM via REST API."""
        # Prepare features
        features_df = pd.DataFrame([
            self._extract_features(lead) for lead in leads
        ])
        
        # Predict scores
        scores = self.model.predict(features_df)
        
        # Prepare batch update
        updates = []
        for lead, score in zip(leads, scores):
            updates.append({
                'id': lead['id'],
                'ai_score': float(score),
                'ai_score_tier': self._get_tier(score),
                'scored_at': datetime.utcnow().isoformat()
            })
        
        # Push in batches
        batch_size = 100
        for i in range(0, len(updates), batch_size):
            batch = updates[i:i + batch_size]
            response = self.session.post(
                f"{self.endpoint}/scores/batch",
                json={'scores': batch}
            )
            response.raise_for_status()
```

---

## 12. Model Monitoring and Retraining

### 12.1 Performance Monitoring

```python
class ModelMonitor:
    """
    Continuous monitoring of model performance in production.
    Detects drift, degradation, and data quality issues.
    """
    
    def __init__(self, model, feature_store, reference_data):
        self.model = model
        self.feature_store = feature_store
        self.reference_data = reference_data
        self.metrics_history = []
        
    def monitor_batch(self, predictions: np.ndarray, actuals: np.ndarray = None,
                      features: pd.DataFrame = None):
        """Monitor a batch of predictions."""
        timestamp = datetime.now()
        
        metrics = {
            'timestamp': timestamp,
            'n_predictions': len(predictions),
            'mean_score': float(predictions.mean()),
            'score_std': float(predictions.std()),
            'score_distribution': self._score_distribution(predictions)
        }
        
        # If we have actuals, calculate performance metrics
        if actuals is not None:
            metrics['auc_roc'] = roc_auc_score(actuals, predictions)
            metrics['actual_conversion_rate'] = float(actuals.mean())
        
        # Feature drift detection
        if features is not None:
            drift_scores = self._detect_feature_drift(features)
            metrics['feature_drift'] = drift_scores
            metrics['max_feature_drift'] = max(drift_scores.values()) if drift_scores else 0
        
        self.metrics_history.append(metrics)
        
        # Alert if anomalies detected
        if self._detect_anomaly(metrics):
            self._send_alert(metrics)
        
        return metrics
    
    def _detect_feature_drift(self, current_features: pd.DataFrame) -> dict:
        """
        Detect drift in feature distributions using 
        Population Stability Index (PSI).
        """
        drift_scores = {}
        
        for col in current_features.columns:
            if col in self.reference_data.columns:
                # Calculate PSI
                psi = self._calculate_psi(
                    self.reference_data[col], 
                    current_features[col]
                )
                if psi > 0.1:  # Significant drift threshold
                    drift_scores[col] = psi
        
        return drift_scores
    
    def _calculate_psi(self, expected, actual, bins=10):
        """Calculate Population Stability Index."""
        # Discretize into bins
        expected_bins = pd.qcut(expected, q=bins, labels=False, duplicates='drop')
        
        # Get bin edges from expected
        _, bin_edges = pd.qcut(expected, q=bins, retbins=True, duplicates='drop')
        
        # Apply same bins to actual
        try:
            actual_binned = pd.cut(actual, bins=bin_edges, labels=False, include_lowest=True)
        except:
            return 0
        
        psi = 0
        for i in range(len(bin_edges) - 1):
            expected_pct = (expected_bins == i).mean() + 1e-6
            actual_pct = (actual_binned == i).mean() + 1e-6
            psi += (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
        
        return psi
    
    def get_retraining_recommendation(self) -> dict:
        """Determine if model needs retraining."""
        recent_metrics = self.metrics_history[-30:]  # Last 30 batches
        
        if len(recent_metrics) < 5:
            return {'needs_retraining': False, 'reason': 'Insufficient data'}
        
        # Check AUC degradation
        if recent_metrics[0].get('auc_roc') and recent_metrics[-1].get('auc_roc'):
            auc_degradation = recent_metrics[0]['auc_roc'] - recent_metrics[-1]['auc_roc']
            if auc_degradation > 0.05:
                return {
                    'needs_retraining': True,
                    'reason': f'AUC decreased by {auc_degradation:.3f}',
                    'severity': 'high'
                }
        
        # Check score drift
        avg_score_trend = [m['mean_score'] for m in recent_metrics]
        if max(avg_score_trend) - min(avg_score_trend) > 0.15:
            return {
                'needs_retraining': True,
                'reason': 'Score distribution has shifted significantly',
                'severity': 'medium'
            }
        
        # Check feature drift
        high_drift_count = sum(
            1 for m in recent_metrics 
            if m.get('max_feature_drift', 0) > 0.2
        )
        if high_drift_count > 5:
            return {
                'needs_retraining': True,
                'reason': f'{high_drift_count} batches with high feature drift',
                'severity': 'medium'
            }
        
        # Time-based retraining
        days_since_training = (datetime.now() - self.model.training_date).days
        if days_since_training > 90:
            return {
                'needs_retraining': True,
                'reason': f'{days_since_training} days since last training',
                'severity': 'low'
            }
        
        return {'needs_retraining': False, 'reason': 'Model performing well'}
    
    def _detect_anomaly(self, metrics: dict) -> bool:
        """Detect anomalous patterns in metrics."""
        # Sudden drop in mean score
        if len(self.metrics_history) > 1:
            prev_mean = self.metrics_history[-2]['mean_score']
            current_mean = metrics['mean_score']
            if abs(current_mean - prev_mean) > 0.1:
                return True
        
        return False
```

### 12.2 Retraining Schedule

| Trigger | Action | Frequency |
|---------|--------|-----------|
| AUC drops > 0.05 | Immediate retraining | As needed |
| Score distribution shifts > 15% | Retrain with recent data | As needed |
| Feature drift detected in 5+ features | Investigate and retrain | As needed |
| 90 days since last training | Scheduled retraining | Quarterly |
| New data sources available | Full retraining | As needed |
| Business model changes | Full retraining | As needed |

---

## 13. Implementation Code

### 13.1 Production Scoring Module

```python
"""
Production-ready lead scoring module.
Optimized for low-latency inference in production.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import json
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
import redis
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LeadScore:
    lead_id: str
    score: float
    tier: str
    components: Dict[str, float]
    model_version: str
    scored_at: str

class ProductionScorer:
    """
    Production lead scoring service.
    Handles feature retrieval, scoring, and result caching.
    """
    
    def __init__(self, model_path: str, config_path: str, redis_url: str = None):
        # Load model
        self.model = xgb.XGBClassifier()
        self.model.load_model(f"{model_path}/model.json")
        
        with open(f"{model_path}/metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        with open(f"{model_path}/features.json", 'r') as f:
            self.feature_config = json.load(f)
        
        self.model_version = self.metadata.get('version', '1.0')
        
        # Connect to feature store
        if redis_url:
            self.feature_store = redis.from_url(redis_url)
        else:
            self.feature_store = None
        
        # Score cache
        self.score_cache = {}
        
        logger.info(f"Model v{self.model_version} loaded. "
                   f"Features: {len(self.feature_config['all_features'])}")
    
    def score_lead(self, lead: Dict) -> LeadScore:
        """Score a single lead."""
        # Extract features
        features = self._extract_features(lead)
        
        # Prepare feature vector
        X = self._prepare_vector(features)
        
        # Predict
        score = float(self.model.predict_proba([X])[0, 1])
        
        # Determine tier
        tier = self._get_tier(score)
        
        # Calculate component scores (for explainability)
        components = self._calculate_components(features)
        
        result = LeadScore(
            lead_id=lead.get('id', 'unknown'),
            score=score,
            tier=tier,
            components=components,
            model_version=self.model_version,
            scored_at=datetime.utcnow().isoformat()
        )
        
        return result
    
    def score_batch(self, leads: List[Dict]) -> List[LeadScore]:
        """Score a batch of leads efficiently."""
        if not leads:
            return []
        
        # Extract features for all leads
        features_list = [self._extract_features(lead) for lead in leads]
        X = np.array([self._prepare_vector(f) for f in features_list])
        
        # Batch predict
        scores = self.model.predict_proba(X)[:, 1]
        
        results = []
        for lead, score in zip(leads, scores):
            results.append(LeadScore(
                lead_id=lead.get('id', 'unknown'),
                score=float(score),
                tier=self._get_tier(float(score)),
                components={},  # Skip components for batch scoring
                model_version=self.model_version,
                scored_at=datetime.utcnow().isoformat()
            ))
        
        return results
    
    def _extract_features(self, lead: Dict) -> Dict:
        """Extract and compute features from lead data."""
        features = {}
        
        for feature in self.feature_config['all_features']:
            # Check direct field
            if feature in lead:
                features[feature] = lead[feature]
            # Check nested field
            elif '.' in feature:
                parts = feature.split('.')
                value = lead
                for part in parts:
                    if isinstance(value, dict):
                        value = value.get(part, 0)
                    else:
                        value = 0
                features[feature] = value
            # Compute derived feature
            elif feature.startswith('days_since_'):
                features[feature] = self._compute_recency(lead, feature)
            else:
                features[feature] = 0
        
        return features
    
    def _compute_recency(self, lead: Dict, feature: str) -> int:
        """Compute days since a specific event."""
        date_field = feature.replace('days_since_', '') + '_date'
        date_value = lead.get(date_field)
        
        if date_value:
            try:
                event_date = pd.to_datetime(date_value)
                return (datetime.now() - event_date).days
            except:
                return 365  # Default: assume long ago
        return 365
    
    def _prepare_vector(self, features: Dict) -> np.ndarray:
        """Convert features dict to numpy array in correct order."""
        return np.array([
            features.get(f, 0) for f in self.feature_config['all_features']
        ])
    
    def _get_tier(self, score: float) -> str:
        """Map score to tier label."""
        if score >= 0.8:
            return 'hot'
        elif score >= 0.5:
            return 'warm'
        elif score >= 0.2:
            return 'cool'
        else:
            return 'cold'
    
    def _calculate_components(self, features: Dict) -> Dict:
        """Calculate component scores for explainability."""
        # Simplified component breakdown
        engagement = sum(features.get(f, 0) for f in 
                        ['email_opens', 'email_clicks', 'website_visits', 
                         'content_downloads', 'pricing_page_views'])
        
        fit = sum(features.get(f, 0) for f in 
                 ['company_size_fit', 'industry_fit', 'tech_fit'])
        
        intent = features.get('intent_score', 0)
        
        recency = 1 / (features.get('days_since_last_activity', 365) + 1)
        
        return {
            'engagement_component': float(min(engagement / 10, 1.0)),
            'fit_component': float(min(fit / 3, 1.0)),
            'intent_component': float(intent),
            'recency_component': float(recency)
        }
```

---

## 14. Tools and Vendors

### 14.1 Commercial Lead Scoring Platforms

| Platform | Key Features | Pricing | Best For |
|----------|-------------|---------|----------|
| **6sense** | Account-based AI, intent data, predictive scoring | Enterprise | ABM programs |
| **Lattice Engines (D&B)** | B2B predictive scoring, firmographic data | Enterprise | Enterprise sales |
| **Everstring** | Account-based lead scoring, intent | Enterprise | ABM |
| **MadKudu** | Custom ML models for product-led growth | Usage-based | PLG companies |
| **InsideSales (Xant)** | Predictive analytics, playbooks | Enterprise | Sales acceleration |
| **Salesforce Einstein** | Native SFDC ML scoring | Included | Salesforce customers |
| **HubSpot Breeze AI** | Native HubSpot ML scoring | Included | HubSpot customers |
| **Gainsight PX** | Product-led scoring | Per-seat | SaaS companies |

### 14.2 Open Source Libraries

- **XGBoost**: Gradient boosting framework (most popular for lead scoring)
- **LightGBM**: Microsoft's gradient boosting (faster training)
- **CatBoost**: Yandex's gradient boosting (handles categorical features well)
- **scikit-learn**: General ML library (baseline models)
- **Optuna**: Hyperparameter optimization
- **SHAP**: Model interpretability
- **MLflow**: ML lifecycle management
- **Feast**: Feature store for ML

---

## 15. Best Practices and Pitfalls

### 15.1 Best Practices

1. **Start Simple**: Begin with logistic regression as baseline, then add complexity
2. **Invest in Feature Engineering**: Quality features > complex models
3. **Monitor for Drift**: Models degrade; continuous monitoring is essential
4. **Calibrate Probabilities**: Ensure scores are interpretable as probabilities
5. **Explain Predictions**: Use SHAP to explain scores to stakeholders
6. **Incorporate Feedback**: Use sales team feedback on lead quality to improve
7. **Retrain Regularly**: At minimum quarterly, or when drift is detected
8. **A/B Test Models**: Compare new models against production via shadow scoring
9. **Handle Missing Data**: Have clear strategy for missing feature values
10. **Document Everything**: Feature definitions, model performance, decisions

### 15.2 Common Pitfalls

| Pitfall | Impact | Solution |
|---------|--------|----------|
| **Label leakage** | Overly optimistic performance | Ensure no future information in features |
| **Ignoring time** | Model learns historical patterns that may not hold | Time-based split, not random |
| **Overfitting** | Poor generalization on new data | Proper validation, regularization |
| **Feature drift** | Model degrades as data distribution shifts | Monitor PSI, retrain regularly |
| **Imbalanced data** | Model predicts majority class only | Class weights, SMOTE, threshold tuning |
| **Actionability gap** | Model scores don't translate to actions | Define clear actions per score tier |
| **Bias propagation** | Model amplifies existing biases | Fairness audit, diverse training data |
| **Ignoring model uncertainty** | Overconfident in predictions | Use probabilistic outputs, confidence intervals |
| **Siloed data** | Missing cross-channel signals | CDP integration for unified data |
| **No feedback loop** | Model can't learn from outcomes | Track actual conversions, use for retraining |

---

*This document is part of the AI Sales & Marketing Knowledge Base. For complementary content, see [01-Overview.md](01-Overview.md), [05-AI-Personalization-and-CDP.md](05-AI-Personalization-and-CDP.md), and [08-AI-Marketing-Analytics-and-Measurement.md](08-AI-Marketing-Analytics-and-Measurement.md).*
