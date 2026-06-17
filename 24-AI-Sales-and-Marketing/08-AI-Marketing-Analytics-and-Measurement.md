# AI Marketing Analytics and Measurement

> **Document Version**: 1.0 — June 2026
> **Scope**: Comprehensive guide to AI-driven marketing analytics — marketing mix modeling (Bayesian MMM with PyMC), multi-touch attribution, customer journey analytics, predictive campaign performance, LTV prediction, experimentation, and measurement frameworks.

---

## Table of Contents

1. [Introduction to AI Marketing Analytics](#1-introduction-to-ai-marketing-analytics)
2. [Marketing Mix Modeling (MMM)](#2-marketing-mix-modeling-mmm)
   - [2.1 Bayesian MMM with PyMC](#21-bayesian-mmm-with-pymc)
   - [2.2 Saturation and Carryover Effects](#22-saturation-and-carryover-effects)
   - [2.3 Media Mix Optimization](#23-media-mix-optimization)
3. [Multi-Touch Attribution](#3-multi-touch-attribution)
   - [3.1 Data-Driven Attribution Models](#31-data-driven-attribution-models)
   - [3.2 Attribution with Markov Chains](#32-attribution-with-markov-chains)
   - [3.3 Shapley Value Attribution](#33-shapley-value-attribution)
4. [Customer Journey Analytics](#4-customer-journey-analytics)
   - [4.1 Journey Mapping with ML](#41-journey-mapping-with-ml)
   - [4.2 Next-Best-Action Prediction](#42-next-best-action-prediction)
   - [4.3 Churn Prediction](#43-churn-prediction)
5. [Predictive Campaign Performance](#5-predictive-campaign-performance)
   - [5.1 Campaign Response Modeling](#51-campaign-response-modeling)
   - [5.2 Budget Optimization](#52-budget-optimization)
6. [Lifetime Value Prediction](#6-lifetime-value-prediction)
   - [6.1 Probabilistic LTV Models](#61-probabilistic-ltv-models)
   - [6.2 Deep Learning LTV](#62-deep-learning-ltv)
7. [Experimentation and Causal Inference](#7-experimentation-and-causal-inference)
   - [7.1 A/B Testing with Bayesian Methods](#71-ab-testing-with-bayesian-methods)
   - [7.2 Causal Impact Analysis](#72-causal-impact-analysis)
8. [Marketing Analytics Architecture](#8-marketing-analytics-architecture)
9. [Implementation Code](#9-implementation-code)
10. [Dashboard and Reporting](#10-dashboard-and-reporting)
11. [Privacy and Compliance in Analytics](#11-privacy-and-compliance-in-analytics)
12. [Future Trends](#12-future-trends)

---

## 1. Introduction to AI Marketing Analytics

### 1.1 The Analytics Transformation

Marketing analytics has evolved from descriptive dashboards to a predictive and prescriptive discipline powered by AI. In 2026, marketing organizations use machine learning, Bayesian inference, and causal modeling to understand what drives growth with unprecedented precision.

### 1.2 Key Capabilities

| Capability | Description | Business Impact |
|---|---|---|
| **Marketing Mix Modeling** | Bayesian models quantifying channel contribution | 10-20% improvement in ROAS |
| **Multi-Touch Attribution** | Scientific credit assignment across channels | 15-30% better budget allocation |
| **Customer Journey Analytics** | End-to-end path analysis with ML | 25-40% increase in conversion rates |
| **LTV Prediction** | Probabilistic lifetime value forecasting | 20-50% improvement in acquisition targeting |
| **Campaign Response Modeling** | Uplift and propensity prediction | 30-60% higher response rates |
| **Experimentation** | Bayesian A/B testing and causal inference | Data-driven decision making |

---

## 2. Marketing Mix Modeling (MMM)

### 2.1 Bayesian MMM with PyMC

Marketing Mix Modeling quantifies the impact of various marketing channels on sales or conversions. Bayesian approaches using PyMC provide uncertainty quantification, incorporation of prior knowledge, and natural handling of sparse data.

```python
# Bayesian Marketing Mix Model using PyMC
import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

class BayesianMMM:
    """Bayesian Marketing Mix Model with saturation and carryover effects.
    
    Model structure:
    Sales_t = α + Σ β_k · f(x_{k,t}, λ_k, s_k) + γ · controls_t + ε_t
    
    where f() applies adstock (carryover) and saturation transformations.
    """
    
    def __init__(self, 
                 channel_cols,
                 control_cols=None,
                 adstock_alpha_prior=0.5,
                 saturation_prior=0.5):
        self.channel_cols = channel_cols
        self.control_cols = control_cols or []
        self.adstock_alpha_prior = adstock_alpha_prior
        self.saturation_prior = saturation_prior
        self.model = None
        self.trace = None
        self.adstock_params = {}
    
    def geometric_adstock(self, x, alpha):
        """Geometric adstock transformation for carryover effects.
        
        Adstock captures the decaying effect of advertising over time:
        y_t = x_t + α · y_{t-1}
        
        where α is the retention rate (0 < α < 1).
        """
        n = len(x)
        y = np.zeros(n)
        y[0] = x[0]
        for t in range(1, n):
            y[t] = x[t] + alpha * y[t-1]
        return y
    
    def logistic_saturation(self, x, lam):
        """Logistic saturation transformation for diminishing returns.
        
        f(x) = (1 - exp(-λ · x)) / (1 + exp(-λ · x))
        
        captures the diminishing marginal returns of media spend.
        """
        return (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))
    
    def hill_saturation(self, x, ec, slope):
        """Hill saturation function (more flexible than logistic).
        
        f(x) = 1 / (1 + (x / EC)^(-slope))
        
        where EC is the inflection point and slope controls curve shape.
        """
        return 1 / (1 + (x / ec) ** (-slope))
    
    def build_model(self, data):
        """Build the Bayesian MMM model.
        
        The model accounts for:
        1. Adstock (carryover) — advertising effects persist over time
        2. Saturation — diminishing returns at high spend levels
        3. Seasonal controls — day-of-week, month, holiday effects
        4. Trend — organic growth baseline
        """
        n_ch = len(self.channel_cols)
        n_obs = len(data)
        
        with pm.Model() as model:
            # --- Priors for adstock parameters ---
            adstock_alphas = pm.Beta(
                'adstock_alpha', 
                alpha=2, beta=3,  # Prior: most channels decay within 1-3 weeks
                shape=n_ch
            )
            
            # --- Priors for saturation parameters ---
            sat_ec = pm.LogNormal(
                'saturation_ec', 
                mu=np.log(data[self.channel_cols].median().values * 0.5),
                sigma=1.0,
                shape=n_ch
            )
            sat_slope = pm.Gamma(
                'saturation_slope',
                alpha=3, beta=2,
                shape=n_ch
            )
            
            # --- Channel coefficients ---
            beta_channels = pm.HalfNormal(
                'beta_channel',
                sigma=0.5,
                shape=n_ch
            )
            
            # --- Base sales ---
            intercept = pm.Normal('intercept', mu=0, sigma=1)
            
            # --- Control variable coefficients ---
            n_ctrl = len(self.control_cols)
            if n_ctrl > 0:
                beta_controls = pm.Normal(
                    'beta_control',
                    mu=0,
                    sigma=0.5,
                    shape=n_ctrl
                )
            
            # --- Transform channels ---
            channel_contributions = []
            
            for i, col in enumerate(self.channel_cols):
                x = data[col].values
                
                # 1. Adstock transformation
                x_adstock = self.geometric_adstock(x, adstock_alphas[i])
                
                # 2. Saturation transformation
                x_saturated = self.hill_saturation(x_adstock, sat_ec[i], sat_slope[i])
                
                # 3. Scale contribution
                contribution = beta_channels[i] * x_saturated
                channel_contributions.append(contribution)
            
            # --- Linear predictor ---
            mu = intercept
            for c in channel_contributions:
                mu = mu + c
            
            # Control variables
            if n_ctrl > 0:
                for i, col in enumerate(self.control_cols):
                    mu = mu + beta_controls[i] * data[col].values
            
            # --- Noise model ---
            sigma = pm.HalfNormal('sigma', sigma=1)
            
            # --- Likelihood ---
            sales = pm.Normal(
                'sales',
                mu=mu,
                sigma=sigma,
                observed=data['sales'].values
            )
            
            self.model = model
            return model
    
    def fit(self, data, draws=2000, tune=2000, chains=4, 
            target_accept=0.95, random_seed=42):
        """Fit the Bayesian MMM using MCMC sampling."""
        self.build_model(data)
        
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                idata_kwargs={"log_likelihood": True}
            )
        
        return self
    
    def summary(self):
        """Print model summary with posterior statistics."""
        summary_df = az.summary(
            self.trace, 
            var_names=['beta_channel', 'adstock_alpha', 
                       'saturation_ec', 'saturation_slope',
                       'intercept', 'sigma']
        )
        return summary_df
    
    def plot_posteriors(self):
        """Plot posterior distributions for all parameters."""
        return az.plot_trace(self.trace, 
                            var_names=['beta_channel', 'adstock_alpha',
                                      'saturation_ec', 'saturation_slope'])
    
    def predict(self, data):
        """Generate posterior predictive samples."""
        with self.model:
            posterior_predictive = pm.sample_posterior_predictive(
                self.trace, 
                random_seed=42
            )
        return posterior_predictive
    
    def get_channel_contributions(self, data, ci=0.94):
        """Compute decomposed channel contributions over time.
        
        Returns the contribution of each channel to sales,
        including uncertainty intervals.
        """
        n_ch = len(self.channel_cols)
        n_obs = len(data)
        
        # Extract posterior samples
        beta_ch = self.trace.posterior['beta_channel'].values
        adstock_alphas = self.trace.posterior['adstock_alpha'].values
        sat_ec = self.trace.posterior['saturation_ec'].values
        sat_slope = self.trace.posterior['saturation_slope'].values
        
        n_samples = beta_ch.shape[0] * beta_ch.shape[1]
        
        # Compute contributions for each posterior sample
        contributions = np.zeros((n_samples, n_obs, n_ch))
        
        idx = 0
        for chain in range(beta_ch.shape[0]):
            for sample in range(beta_ch.shape[1]):
                for i in range(n_ch):
                    x = data[self.channel_cols[i]].values
                    x_ad = self.geometric_adstock(x, adstock_alphas[chain, sample, i])
                    x_sat = self.hill_saturation(x_ad, sat_ec[chain, sample, i], 
                                                  sat_slope[chain, sample, i])
                    contributions[idx, :, i] = beta_ch[chain, sample, i] * x_sat
                idx += 1
        
        # Aggregate to mean and credible intervals
        result = {}
        for i, col in enumerate(self.channel_cols):
            contrib = contributions[:, :, i]
            result[col] = {
                'mean': contrib.mean(axis=0),
                'lower': np.percentile(contrib, (1 - ci) / 2 * 100, axis=0),
                'upper': np.percentile(contrib, (1 + ci) / 2 * 100, axis=0),
                'total': contrib.mean(axis=0).sum(),
                'total_lower': np.percentile(contrib.sum(axis=1), (1 - ci) / 2 * 100),
                'total_upper': np.percentile(contrib.sum(axis=1), (1 + ci) / 2 * 100)
            }
        
        return result
    
    def get_roi(self, data, ci=0.94):
        """Compute ROI with uncertainty for each channel.
        
        ROI_k = Total Contribution_k / Total Spend_k
        """
        contributions = self.get_channel_contributions(data, ci)
        
        roi_data = {}
        for col in self.channel_cols:
            total_spend = data[col].sum()
            contrib = contributions[col]
            
            roi_data[col] = {
                'roi_mean': contrib['total'] / total_spend,
                'roi_lower': contrib['total_lower'] / total_spend,
                'roi_upper': contrib['total_upper'] / total_spend,
                'total_contribution': contrib['total'],
                'total_spend': total_spend,
                'contribution_pct': contrib['total'] / sum(
                    contributions[c]['total'] for c in self.channel_cols
                ) * 100
            }
        
        return pd.DataFrame(roi_data).T

### 2.2 Example: Running the Bayesian MMM

```python
# Full example of Bayesian MMM in practice
def run_mmm_example():
    """Run a complete MMM analysis with synthetic data."""
    
    # Generate synthetic weekly data (2 years)
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=104, freq='W')
    
    data = pd.DataFrame({'date': dates})
    
    # Channel spends with realistic patterns
    data['tv_spend'] = np.random.uniform(5000, 20000, 104) + \
                       5000 * np.sin(2 * np.pi * np.arange(104) / 52)
    data['digital_spend'] = np.random.uniform(3000, 15000, 104) + \
                            3000 * np.sin(2 * np.pi * np.arange(104) / 26)
    data['social_spend'] = np.random.uniform(1000, 8000, 104)
    data['search_spend'] = np.random.uniform(2000, 10000, 104)
    
    # Generate sales with known contributions
    base_sales = 50000
    tv_effect = 2.5 * (1 - np.exp(-0.0001 * data['tv_spend'].values))
    digital_effect = 3.0 * (1 - np.exp(-0.0002 * data['digital_spend'].values))
    social_effect = 1.5 * (1 - np.exp(-0.0003 * data['social_spend'].values))
    search_effect = 4.0 * (1 - np.exp(-0.0005 * data['search_spend'].values))
    
    # Add adstock carryover effects
    def apply_adstock(x, alpha=0.7):
        y = np.zeros_like(x)
        y[0] = x[0]
        for t in range(1, len(x)):
            y[t] = x[t] + alpha * y[t-1]
        return y
    
    tv_effect = apply_adstock(tv_effect, 0.8)
    digital_effect = apply_adstock(digital_effect, 0.6)
    social_effect = apply_adstock(social_effect, 0.4)
    search_effect = apply_adstock(search_effect, 0.5)
    
    # Add seasonality
    seasonality = 1 + 0.2 * np.sin(2 * np.pi * np.arange(104) / 52)
    
    # Final sales
    noise = np.random.normal(0, 3000, 104)
    data['sales'] = (base_sales + tv_effect + digital_effect + 
                     social_effect + search_effect) * seasonality + noise
    
    # Add control variables
    data['holiday'] = (data['date'].dt.month == 12).astype(float)
    data['competitor_spend'] = np.random.uniform(10000, 30000, 104)
    
    # Fit Bayesian MMM
    mmm = BayesianMMM(
        channel_cols=['tv_spend', 'digital_spend', 'social_spend', 'search_spend'],
        control_cols=['holiday', 'competitor_spend']
    )
    
    mmm.fit(data, draws=1000, tune=1000, chains=2)
    
    # Results
    print("=== Model Summary ===")
    print(mmm.summary())
    
    print("\n=== ROI Analysis ===")
    roi = mmm.get_roi(data)
    print(roi)
    
    print("\n=== Channel Contribution Breakdown ===")
    contributions = mmm.get_channel_contributions(data)
    for ch, vals in contributions.items():
        print(f"{ch}: {vals['total']:.0f} total contribution "
              f"[{vals['total_lower']:.0f}, {vals['total_upper']:.0f}]")
    
    return mmm, data

# Uncomment to run:
# mmm, data = run_mmm_example()
```

### 2.3 Saturation and Carryover Effects Deep Dive

#### Carryover (Adstock) Models

| Model | Formula | Description | Use Case |
|---|---|---|---|
| **Geometric** | y_t = x_t + α · y_{t-1} | Simple exponential decay | Most common for advertising |
| **Weibull** | y_t = Σ x_{t-i} · f(i; λ, k) | Flexible decay shapes | When peak effect is delayed |
| **Delayed** | y_t = α · y_{t-1} + β · x_{t-d} | Explicit lag period | Direct response campaigns |
| **Video** | y_t = Σ x_{t-i} · w_i | Non-parametric weights | Complex media mixes |

#### Saturation Models

| Model | Formula | Inflection Point | Flexibility |
|---|---|---|---|
| **Logistic** | f(x) = L / (1 + e^{-k(x-x₀)}) | Fixed at midpoint | Moderate |
| **Hill** | f(x) = 1 / (1 + (x/EC)^{-slope}) | Controllable via EC | High |
| **Michaelis-Menten** | f(x) = V_max · x / (K_m + x) | At K_m | Low |
| **Adbudg** | f(x) = x^β / (1 + γ · x^β) | Flexible saturation | Very High |

```python
# Comparative visualization of saturation curves
def plot_saturation_curves():
    """Plot different saturation functions for comparison."""
    x = np.linspace(0, 100, 500)
    
    plt.figure(figsize=(12, 6))
    
    # Logistic saturation
    for lam in [0.05, 0.1, 0.2]:
        y = (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))
        plt.plot(x, y, label=f'Logistic λ={lam}', linestyle='--')
    
    # Hill saturation
    for ec, slope in [(20, 2), (30, 3), (50, 4)]:
        y = 1 / (1 + (x / ec) ** (-slope))
        plt.plot(x, y, label=f'Hill EC={ec}, s={slope}')
    
    plt.xlabel('Spend ($K)')
    plt.ylabel('Saturation Level')
    plt.title('Ad Spend Saturation Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('saturation_curves.png', dpi=150)
    plt.close()
```

### 2.4 Media Mix Optimization

```python
class MediaOptimizer:
    """Optimize media budget allocation using the fitted MMM.
    
    Given the Bayesian MMM posterior, find the optimal budget
    allocation that maximizes ROI or sales under budget constraints.
    """
    
    def __init__(self, mmm_model, data):
        self.mmm = mmm_model
        self.data = data
        self.channels = mmm_model.channel_cols
        self.n_channels = len(self.channels)
    
    def optimize_budget(self, total_budget, objective='sales', 
                        min_alloc=0.05, max_alloc=0.6):
        """Find optimal budget allocation using Bayesian optimization.
        
        Args:
            total_budget: Total weekly budget to allocate
            objective: 'sales' or 'roi' — what to maximize
            min_alloc: Minimum fraction of budget per channel
            max_alloc: Maximum fraction of budget per channel
        
        Returns:
            Optimal allocation dict with uncertainty
        """
        from scipy.optimize import minimize
        
        # Objective function: negative expected sales (for minimization)
        def negative_expected_sales(weights):
            weights = np.clip(weights, min_alloc, max_alloc)
            weights = weights / weights.sum()  # Ensure sum to 1
            budgets = weights * total_budget
            
            # Use posterior mean to estimate sales contribution
            total_contrib = 0
            
            # Extract posterior means
            beta_mean = self.mmm.trace.posterior['beta_channel'].mean(
                dim=['chain', 'draw']
            ).values
            ec_mean = self.mmm.trace.posterior['saturation_ec'].mean(
                dim=['chain', 'draw']
            ).values
            slope_mean = self.mmm.trace.posterior['saturation_slope'].mean(
                dim=['chain', 'draw']
            ).values
            alpha_mean = self.mmm.trace.posterior['adstock_alpha'].mean(
                dim=['chain', 'draw']
            ).values
            
            for i, ch in enumerate(self.channels):
                # For a sustained allocation, the steady-state contribution
                # accounts for adstock accumulation
                steady_spend = budgets[i]
                saturated = self.mmm.hill_saturation(
                    steady_spend, ec_mean[i], slope_mean[i]
                )
                total_contrib += beta_mean[i] * saturated
            
            return -(total_contrib + self.mmm.trace.posterior['intercept'].mean().values)
        
        # Initial guess: equal allocation
        x0 = np.ones(self.n_channels) / self.n_channels
        
        # Constraints: sum to 1, bounds per channel
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        bounds = [(min_alloc, max_alloc) for _ in range(self.n_channels)]
        
        result = minimize(
            negative_expected_sales,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        optimal_weights = result.x
        optimal_weights = np.clip(optimal_weights, min_alloc, max_alloc)
        optimal_weights = optimal_weights / optimal_weights.sum()
        
        # Compute uncertainty via posterior sampling
        allocations = []
        n_samples = 500
        
        for _ in range(n_samples):
            # Sample from posterior
            idx_chain = np.random.randint(0, self.mmm.trace.posterior.dims['chain'])
            idx_draw = np.random.randint(0, self.mmm.trace.posterior.dims['draw'])
            
            beta_s = self.mmm.trace.posterior['beta_channel'].values[
                idx_chain, idx_draw
            ]
            ec_s = self.mmm.trace.posterior['saturation_ec'].values[
                idx_chain, idx_draw
            ]
            slope_s = self.mmm.trace.posterior['saturation_slope'].values[
                idx_chain, idx_draw
            ]
            
            expected_sales = 0
            for i, ch in enumerate(self.channels):
                steady = (optimal_weights[i] * total_budget)
                sat = self.mmm.hill_saturation(steady, ec_s[i], slope_s[i])
                expected_sales += beta_s[i] * sat
            
            allocations.append(expected_sales)
        
        return {
            'optimal_allocation': {
                self.channels[i]: {
                    'fraction': optimal_weights[i],
                    'budget': optimal_weights[i] * total_budget
                }
                for i in range(self.n_channels)
            },
            'expected_weekly_sales': np.mean(allocations),
            'sales_ci': np.percentile(allocations, [3, 97]),
            'improvement_vs_current': (
                np.mean(allocations) / self._current_expected_sales(total_budget) - 1
            ) * 100
        }
    
    def _current_expected_sales(self, total_budget):
        """Expected sales at current allocation."""
        current_weights = np.array([
            self.data[ch].mean() for ch in self.channels
        ])
        current_weights = current_weights / current_weights.sum()
        return -self.negative_expected_sales(current_weights)

---

## 3. Multi-Touch Attribution

### 3.1 Data-Driven Attribution Models

Multi-touch attribution (MTA) assigns credit for conversions across all touchpoints in the customer journey, not just the last click. Machine learning enables data-driven attribution that goes beyond heuristic rules.

```python
# Data-Driven Attribution Framework
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

class DataDrivenAttribution:
    """Data-driven multi-touch attribution using ML.
    
    Approaches:
    1. Logistic regression: interpretable, industry standard
    2. Random Forest: handles interactions, non-linearities
    3. Custom transformer: builds attention-weighted attribution
    """
    
    def __init__(self, channels, method='logistic'):
        self.channels = channels
        self.method = method
        self.model = None
        self.channel_weights = None
    
    def _build_transition_features(self, journeys):
        """Build feature matrix from customer journeys.
        
        Each journey becomes a feature vector where each element
        represents the presence, count, or position-weighted
        importance of a channel.
        """
        features = []
        labels = []
        
        for journey in journeys:
            touches = journey['touches']
            converted = journey['converted']
            
            # Feature: channel presence (binary)
            presence = {ch: 0 for ch in self.channels}
            
            # Feature: channel counts
            counts = {ch: 0 for ch in self.channels}
            
            # Feature: position-weighted (first click, last click bonus)
            position_weighted = {ch: 0.0 for ch in self.channels}
            
            n_touches = len(touches)
            for i, ch in enumerate(touches):
                if ch in presence:
                    presence[ch] = 1
                    counts[ch] += 1
                    # Position weight: linear decay from 1 to 0.5
                    position_weighted[ch] += 1 - (i / n_touches) * 0.5
            
            # Normalize position weights by touch count
            if n_touches > 0:
                position_weighted = {k: v / n_touches for k, v in position_weighted.items()}
            
            # Feature: time decay (exponential)
            time_decay = {ch: 0.0 for ch in self.channels}
            if n_touches > 0:
                for i, ch in enumerate(touches):
                    if ch in time_decay:
                        time_decay[ch] += np.exp(-(n_touches - 1 - i) * 0.5)
                time_decay = {k: v / n_touches for k, v in time_decay.items()}
            
            # Combine all features
            row = []
            for ch in self.channels:
                row.extend([
                    presence[ch],
                    counts[ch] / max(n_touches, 1),
                    position_weighted[ch],
                    time_decay[ch]
                ])
            
            # Additional features
            row.append(n_touches)  # Total touchpoints
            row.append(1 if n_touches == 1 else 0)  # Single touch?
            
            features.append(row)
            labels.append(1 if converted else 0)
        
        return np.array(features), np.array(labels)
    
    def fit(self, journeys):
        """Fit attribution model on historical journey data."""
        X, y = self._build_transition_features(journeys)
        
        if self.method == 'logistic':
            self.model = LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                C=0.5,
                random_state=42
            )
        elif self.method == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                class_weight='balanced',
                random_state=42
            )
        elif self.method == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                random_state=42
            )
        
        self.model.fit(X, y)
        
        # Extract channel importance
        self._compute_channel_weights()
        
        return self
    
    def _compute_channel_weights(self):
        """Compute overall channel importance from model coefficients.
        
        For logistic regression: use coefficient magnitude.
        For tree-based: use feature importance aggregated by channel.
        """
        if self.method == 'logistic':
            # Each channel has 4 features; aggregate their coefficients
            coefs = self.model.coef_[0]
            self.channel_weights = {}
            for i, ch in enumerate(self.channels):
                start = i * 4
                channel_coefs = coefs[start:start+4]
                self.channel_weights[ch] = np.sum(np.abs(channel_coefs))
        
        elif self.method in ['random_forest', 'gradient_boosting']:
            importances = self.model.feature_importances_
            self.channel_weights = {}
            for i, ch in enumerate(self.channels):
                start = i * 4
                ch_importance = importances[start:start+4].sum()
                self.channel_weights[ch] = ch_importance
        
        # Normalize to attribution percentages
        total = sum(self.channel_weights.values())
        if total > 0:
            for ch in self.channel_weights:
                self.channel_weights[ch] /= total
    
    def attribute(self, journeys):
        """Attribute conversions across channels using fitted model."""
        if self.channel_weights is None:
            raise ValueError("Model must be fitted first")
        
        total_conversions = sum(1 for j in journeys if j['converted'])
        
        return {
            ch: {
                'weight': self.channel_weights[ch],
                'attributed_conversions': self.channel_weights[ch] * total_conversions
            }
            for ch in self.channels
        }
    
    def get_journey_attribution(self, journey):
        """Attribute a single journey across channels."""
        X, _ = self._build_transition_features([journey])
        
        # Get per-touchpoint contribution
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)[0]
            conversion_prob = probabilities[1]
        else:
            conversion_prob = self.model.predict(X)[0]
        
        # Distribute credit based on position weights
        n_touches = len(journey['touches'])
        attribution = {}
        
        for i, ch in enumerate(journey['touches']):
            if ch not in attribution:
                attribution[ch] = 0
            # Weight by position (higher weight for earlier touches)
            attribution[ch] += (1 - i / n_touches) / n_touches
        
        # Normalize
        total = sum(attribution.values())
        if total > 0:
            for ch in attribution:
                attribution[ch] = (attribution[ch] / total) * conversion_prob
        
        return {
            'conversion_probability': conversion_prob,
            'attribution': attribution
        }
```

### 3.2 Attribution with Markov Chains

```python
class MarkovChainAttribution:
    """First-order Markov chain attribution model.
    
    Models the customer journey as a stochastic process where
    states represent marketing channels plus START, CONVERT, and DROP.
    Computes channel importance via removal effect.
    """
    
    def __init__(self, channels):
        self.channels = channels
        self.states = ['START'] + channels + ['CONVERT', 'NULL']
        self.n_states = len(self.states)
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.transition_matrix = None
    
    def fit(self, journeys):
        """Estimate transition probabilities from journey data."""
        # Count transitions
        counts = np.zeros((self.n_states, self.n_states))
        
        for journey in journeys:
            touches = journey['touches']
            converted = journey['converted']
            
            # Start -> first channel
            first_ch = touches[0] if touches else 'NULL'
            if first_ch in self.state_to_idx:
                counts[self.state_to_idx['START'], 
                       self.state_to_idx[first_ch]] += 1
            
            # Channel-to-channel transitions
            for i in range(len(touches) - 1):
                from_ch = touches[i]
                to_ch = touches[i + 1]
                if from_ch in self.state_to_idx and to_ch in self.state_to_idx:
                    counts[self.state_to_idx[from_ch], 
                           self.state_to_idx[to_ch]] += 1
            
            # Last channel -> outcome
            if touches:
                last_ch = touches[-1]
                if last_ch in self.state_to_idx:
                    outcome = 'CONVERT' if converted else 'NULL'
                    counts[self.state_to_idx[last_ch], 
                           self.state_to_idx[outcome]] += 1
        
        # Convert to probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        self.transition_matrix = counts / row_sums
        
        return self
    
    def _absorption_probability(self, transition_matrix):
        """Compute probability of reaching CONVERT from each state."""
        convert_idx = self.state_to_idx['CONVERT']
        null_idx = self.state_to_idx['NULL']
        
        # Identify transient states (all except CONVERT and NULL)
        transient = [i for i in range(self.n_states) 
                    if i not in [convert_idx, null_idx]]
        n_transient = len(transient)
        
        if n_transient == 0:
            return np.array([0.0])
        
        # Q: transitions among transient states
        Q = transition_matrix[np.ix_(transient, transient)]
        
        # R: transitions from transient to absorbing states
        R = transition_matrix[np.ix_(transient, [convert_idx])]
        
        # Fundamental matrix: N = (I - Q)^(-1)
        I = np.eye(n_transient)
        N = np.linalg.inv(I - Q)
        
        # Absorption probabilities: B = N @ R
        B = N @ R
        
        return B.flatten()
    
    def _removal_effect(self, channel, journeys):
        """Compute the effect of removing a channel on total conversions.
        
        The removal effect measures how many conversions would be lost
        if the channel were unavailable.
        """
        # Remove all touches from this channel
        modified_journeys = []
        for journey in journeys:
            modified_touches = [t for t in journey['touches'] if t != channel]
            modified_journeys.append({
                'touches': modified_touches,
                'converted': journey['converted']
            })
        
        # Fit model without channel
        alt_counts = np.zeros((self.n_states, self.n_states))
        for journey in modified_journeys:
            touches = journey['touches']
            converted = journey['converted']
            
            if not touches:
                # No touches -> directly to outcome
                outcome_idx = self.state_to_idx['CONVERT' if converted else 'NULL']
                alt_counts[self.state_to_idx['START'], outcome_idx] += 1
                continue
            
            first_ch = touches[0]
            if first_ch in self.state_to_idx:
                alt_counts[self.state_to_idx['START'], 
                          self.state_to_idx[first_ch]] += 1
            
            for i in range(len(touches) - 1):
                from_ch = touches[i]
                to_ch = touches[i + 1]
                if from_ch in self.state_to_idx and to_ch in self.state_to_idx:
                    alt_counts[self.state_to_idx[from_ch], 
                              self.state_to_idx[to_ch]] += 1
            
            if touches:
                last_ch = touches[-1]
                if last_ch in self.state_to_idx:
                    outcome = 'CONVERT' if converted else 'NULL'
                    alt_counts[self.state_to_idx[last_ch], 
                              self.state_to_idx[outcome]] += 1
        
        # Normalize
        row_sums = alt_counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        alt_transition = alt_counts / row_sums
        
        # Probability of converting without this channel
        convert_probs = self._absorption_probability(alt_transition)
        start_idx_in_transient = list(range(self.n_states - 2)).index(
            self.state_to_idx['START']
        ) if self.state_to_idx['START'] < self.n_states - 2 else 0
        prob_without = convert_probs[start_idx_in_transient]
        
        # Original probability of converting
        original_probs = self._absorption_probability(self.transition_matrix)
        prob_original = original_probs[start_idx_in_transient]
        
        # Removal effect: relative decrease in conversion probability
        if prob_original > 0:
            return (prob_original - prob_without) / prob_original
        return 0
    
    def attribute(self, journeys):
        """Compute Markov chain attribution for all channels."""
        if self.transition_matrix is None:
            self.fit(journeys)
        
        # Total conversions
        total_conversions = sum(1 for j in journeys if j['converted'])
        
        # Compute removal effect for each channel
        effects = {}
        for ch in self.channels:
            effects[ch] = self._removal_effect(ch, journeys)
        
        # Normalize to attribution weights
        total_effect = sum(effects.values())
        if total_effect > 0:
            for ch in effects:
                effects[ch] /= total_effect
        
        return {
            ch: {
                'removal_effect': effects[ch],
                'attributed_conversions': effects[ch] * total_conversions,
                'attribution_pct': effects[ch] * 100
            }
            for ch in self.channels
        }
```

### 3.3 Shapley Value Attribution

```python
class ShapleyAttribution:
    """Game-theoretic attribution using Shapley values.
    
    Each channel is a 'player' in a cooperative game.
    The Shapley value fairly distributes the total conversions
    based on each channel's marginal contribution.
    """
    
    def __init__(self, channels):
        self.channels = channels
        self.n = len(channels)
    
    def _value_function(self, coalition_mask, journeys):
        """Compute conversion value for a coalition of channels.
        
        The value v(S) is the number of conversions that occur
        when only channels in coalition S are present.
        """
        active = {self.channels[i] for i in range(self.n) if coalition_mask[i]}
        
        if not active:
            return 0
        
        conversions = 0
        for j in journeys:
            touches = set(j['touches'])
            # A journey is attributed to the coalition if ALL touches
            # occurred within the active channels
            if touches.issubset(active):
                if j['converted']:
                    conversions += 1
        
        return conversions
    
    def compute(self, journeys):
        """Compute Shapley values for all channels.
        
        φ_i = Σ_{S ⊆ N\{i}} [|S|! (|N|-|S|-1)! / |N|!] × [v(S ∪ {i}) - v(S)]
        """
        from itertools import combinations
        import math
        
        shapley = {ch: 0.0 for ch in self.channels}
        
        for i, channel in enumerate(self.channels):
            other_indices = list(range(self.n))
            other_indices.remove(i)
            
            for r in range(self.n):
                for subset in combinations(other_indices, r):
                    # Coalition without i
                    mask_without = [False] * self.n
                    for idx in subset:
                        mask_without[idx] = True
                    
                    # Coalition with i
                    mask_with = mask_without.copy()
                    mask_with[i] = True
                    
                    # Marginal contribution
                    v_without = self._value_function(mask_without, journeys)
                    v_with = self._value_function(mask_with, journeys)
                    marginal = v_with - v_without
                    
                    # Weight
                    weight = (math.factorial(r) * 
                             math.factorial(self.n - r - 1) / 
                             math.factorial(self.n))
                    
                    shapley[channel] += weight * marginal
        
        # Normalize to total conversions
        total_conv = sum(1 for j in journeys if j['converted'])
        total_shapley = sum(shapley.values())
        
        if total_shapley > 0:
            for ch in shapley:
                shapley[ch] = (shapley[ch] / total_shapley) * total_conv
        
        return shapley
    
    def compare_models(self, journeys):
        """Compare Shapley, Markov, and heuristic attribution."""
        shapley_results = self.compute(journeys)
        
        # Last-click baseline
        last_click = {ch: 0 for ch in self.channels}
        for j in journeys:
            if j['converted'] and j['touches']:
                last_click[j['touches'][-1]] = last_click.get(
                    j['touches'][-1], 0
                ) + 1
        
        # First-click baseline
        first_click = {ch: 0 for ch in self.channels}
        for j in journeys:
            if j['converted'] and j['touches']:
                first_click[j['touches'][0]] = first_click.get(
                    j['touches'][0], 0
                ) + 1
        
        comparison = pd.DataFrame({
            'channel': self.channels,
            'shapley': [shapley_results.get(ch, 0) for ch in self.channels],
            'last_click': [last_click.get(ch, 0) for ch in self.channels],
            'first_click': [first_click.get(ch, 0) for ch in self.channels]
        })
        
        return comparison

---

## 4. Customer Journey Analytics

### 4.1 Journey Mapping with ML

```python
# ML-driven customer journey analysis
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class CustomerJourneyAnalyzer:
    """Analyze and cluster customer journeys using ML.
    
    Identifies common journey patterns, drop-off points,
    and high-value paths to conversion.
    """
    
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.clusterer = None
        self.pca = None
        self.scaler = None
    
    def _encode_journeys(self, journeys):
        """Encode journeys as fixed-length feature vectors.
        
        Features include:
        - Channel sequence encoding (bag-of-channels + position)
        - Time between touches
        - Total touchpoints
        - Channel transition frequencies
        """
        features = []
        
        for j in journeys:
            touches = j['touches']
            n = len(touches)
            
            # Bag of channels
            channel_counts = {ch: 0 for ch in set(touches)}
            for t in touches:
                if t in channel_counts:
                    channel_counts[t] += 1
            
            # Normalized counts
            counts = [channel_counts.get(ch, 0) / max(n, 1) 
                     for ch in sorted(channel_counts.keys())]
            
            # Journey length and complexity
            length_feat = [min(n / 20, 1.0)]  # Normalized
            unique_channels = [len(set(touches)) / max(len(channel_counts), 1)]
            
            # Transition entropy (measure of path randomness)
            if n > 1:
                transitions = [(touches[i], touches[i+1]) 
                              for i in range(n-1)]
                unique_trans = len(set(transitions))
                entropy = unique_trans / max(n - 1, 1)
            else:
                entropy = 0
            entropy_feat = [entropy]
            
            # Recency (last touch -> conversion time)
            time_feat = [j.get('time_to_convert', 0) / 30.0]  # Normalized to month
            
            row = counts + length_feat + unique_channels + entropy_feat + time_feat
            features.append(row)
        
        # Pad to consistent length
        max_len = max(len(f) for f in features) if features else 0
        padded = np.zeros((len(features), max_len))
        for i, f in enumerate(features):
            padded[i, :len(f)] = f
        
        return padded
    
    def fit(self, journeys):
        """Cluster customer journeys to identify common patterns."""
        X = self._encode_journeys(journeys)
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.pca = PCA(n_components=min(10, X_scaled.shape[1]))
        X_pca = self.pca.fit_transform(X_scaled)
        
        self.clusterer = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10
        )
        labels = self.clusterer.fit_predict(X_pca)
        
        # Build cluster profiles
        cluster_profiles = {}
        for cluster_id in range(self.n_clusters):
            cluster_journeys = [j for i, j in enumerate(journeys) 
                               if labels[i] == cluster_id]
            
            if not cluster_journeys:
                continue
            
            all_touches = [t for j in cluster_journeys for t in j['touches']]
            conversion_rate = sum(1 for j in cluster_journeys if j['converted']) / len(cluster_journeys)
            avg_touches = np.mean([len(j['touches']) for j in cluster_journeys])
            
            # Most common channels
            from collections import Counter
            channel_freq = Counter(all_touches).most_common(5)
            
            cluster_profiles[cluster_id] = {
                'size': len(cluster_journeys),
                'conversion_rate': conversion_rate,
                'avg_touchpoints': avg_touches,
                'top_channels': channel_freq,
                'avg_time_to_convert': np.mean([
                    j.get('time_to_convert', 0) for j in cluster_journeys
                ])
            }
        
        return {
            'labels': labels,
            'profiles': cluster_profiles,
            'explained_variance': self.pca.explained_variance_ratio_.cumsum()[-1]
        }
    
    def predict_journey_cluster(self, journey):
        """Assign a new journey to an existing cluster."""
        X = self._encode_journeys([journey])
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return self.clusterer.predict(X_pca)[0]
    
    def get_transition_graph(self, journeys):
        """Build a transition probability graph for visualization."""
        import networkx as nx
        
        G = nx.DiGraph()
        
        # Count transitions
        transitions = {}
        for j in journeys:
            touches = j['touches']
            converted = j['converted']
            
            # Start -> first touch
            if touches:
                start_key = ('START', touches[0])
                transitions[start_key] = transitions.get(start_key, 0) + 1
            
            # Between touches
            for i in range(len(touches) - 1):
                key = (touches[i], touches[i+1])
                transitions[key] = transitions.get(key, 0) + 1
            
            # Last touch -> outcome
            if touches:
                outcome = 'CONVERT' if converted else 'DROP'
                key = (touches[-1], outcome)
                transitions[key] = transitions.get(key, 0) + 1
        
        # Add nodes and edges
        for (src, dst), count in transitions.items():
            G.add_edge(src, dst, weight=count)
        
        return G
```

### 4.2 Next-Best-Action Prediction

```python
class NextBestAction:
    """Predict the optimal next action for each customer.
    
    Uses multi-class classification to recommend the channel
    and message most likely to drive conversion.
    """
    
    def __init__(self):
        self.model = None
        self.action_options = None
        self.feature_columns = None
    
    def fit(self, customer_data, actions, outcomes):
        """Train next-best-action model.
        
        Args:
            customer_data: DataFrame of customer features
            actions: Array of action labels taken
            outcomes: Binary outcome (converted or not)
        """
        from sklearn.ensemble import GradientBoostingClassifier
        
        self.action_options = list(set(actions))
        
        # Feature: customer profile + action taken
        X = customer_data.copy()
        X['action_encoded'] = [self.action_options.index(a) if a in self.action_options 
                               else -1 for a in actions]
        
        self.feature_columns = X.columns.tolist()
        y = outcomes
        
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )
        self.model.fit(X, y)
        
        return self
    
    def recommend(self, customer_features, top_k=3):
        """Recommend top-K actions for a customer."""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        recommendations = []
        
        for action in self.action_options:
            X = customer_features.copy()
            if isinstance(X, pd.Series):
                X = X.to_frame().T
            
            X['action_encoded'] = self.action_options.index(action)
            X = X[self.feature_columns]
            
            prob = self.model.predict_proba(X)[0, 1]
            recommendations.append({
                'action': action,
                'conversion_probability': prob
            })
        
        recommendations.sort(key=lambda x: x['conversion_probability'], reverse=True)
        return recommendations[:top_k]
```

### 4.3 Churn Prediction

```python
class ChurnPredictor:
    """Predict customer churn using behavioral and engagement features.
    
    Combines survival analysis with ML classification
    to predict both churn probability and expected timing.
    """
    
    def __init__(self):
        self.classifier = None
        self.feature_columns = None
    
    def _extract_features(self, customer_data):
        """Extract churn prediction features."""
        features = pd.DataFrame(index=customer_data.index)
        
        # Recency, Frequency, Monetary (RFM)
        features['recency_days'] = customer_data['days_since_last_purchase']
        features['frequency_30d'] = customer_data['purchases_30d']
        features['frequency_90d'] = customer_data['purchases_90d']
        features['monetary_30d'] = customer_data['revenue_30d']
        features['monetary_avg'] = customer_data['avg_order_value']
        
        # Engagement metrics
        features['login_frequency_30d'] = customer_data['logins_30d']
        features['support_tickets_30d'] = customer_data['support_tickets']
        features['email_click_rate'] = customer_data['email_click_rate']
        features['app_session_duration'] = customer_data['avg_session_minutes']
        
        # Product usage
        features['features_used'] = customer_data['num_features_used']
        features['feature_adoption_score'] = customer_data['adoption_score']
        
        # Tenure and lifecycle
        features['tenure_days'] = customer_data['tenure_days']
        features['is_new_customer'] = (customer_data['tenure_days'] < 90).astype(float)
        features['is_at_risk_segment'] = customer_data['risk_segment'].map(
            {'low': 0, 'medium': 0.5, 'high': 1.0}
        )
        
        # Service quality
        features['nps_score'] = customer_data['nps_score'] / 10.0
        features['avg_resolution_time'] = customer_data['avg_resolution_hours'] / 72.0
        
        self.feature_columns = features.columns.tolist()
        return features
    
    def fit(self, customer_data, churn_labels):
        """Train churn prediction model."""
        X = self._extract_features(customer_data)
        y = churn_labels
        
        from xgboost import XGBClassifier
        
        self.classifier = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
            random_state=42,
            eval_metric='auc'
        )
        
        self.classifier.fit(
            X, y,
            eval_set=[(X, y)],
            verbose=False
        )
        
        return self
    
    def predict(self, customer_data):
        """Predict churn probability for customers."""
        X = self._extract_features(customer_data)
        prob = self.classifier.predict_proba(X)[:, 1]
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return prob, importance
    
    def segment_risk(self, customer_data):
        """Segment customers into risk categories."""
        prob, _ = self.predict(customer_data)
        
        segments = pd.cut(
            prob,
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        return {
            'probability': prob,
            'segment': segments,
            'high_risk_pct': (prob > 0.6).mean() * 100
        }
