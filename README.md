# Investor Emotions and Stock Market Behavior

A research project examining the relationship between investor emotions expressed on social media and intraday stock returns using NLP and econometric panel regression methods.



---

## 📊 Research Objective

This project investigates whether investor emotions expressed on social media before market open can predict open-to-close stock returns on the same trading day. By combining natural language processing with econometric analysis, we uncover potential predictive patterns in sentiment-driven trading behavior.

**Key Question**: Can pre-market emotional signals from social media forecast intraday stock price movements?

---

## 🔬 Methodology

### 1. Emotion Extraction
- Social media messages are processed using a pretrained Transformer-based emotion classification model
- Each message is mapped to probability distributions over multiple emotion categories
- Captures nuanced emotional content beyond simple positive/negative sentiment

### 2. Signal Aggregation
- Message-level emotions are aggregated to firm-day level signals
- User influence is incorporated through follower-based weighting
- Time filtering ensures only pre-market messages are included

### 3. Econometric Analysis
- Intraday stock returns (open-to-close) modeled using fixed-effects panel regressions
- **Firm fixed effects**: Control for time-invariant firm characteristics
- **Time fixed effects**: Control for market-wide factors
- **Controls**: Trading volume, volatility, lagged returns
- **Standard errors**: Clustered at the firm level for robustness

---

## 📁 Repository Structure

```
.
├── src/
│   ├── step3_emotion_extraction.py    # Extract emotions from raw messages
│   ├── step4_aggregate_emotions.py    # Aggregate to firm-day level
│   └── step6A_panel_regression.py     # Run panel regressions
├── data/
│   ├── raw/                           # Raw social media messages
│   ├── processed/                     # Emotion-labeled messages
│   └── aggregated/                    # Firm-day level datasets
├── results/
│   ├── tables/                        # Regression output tables
│   └── figures/                       # Visualizations
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**:
```bash
git clone https://github.com/bhanup23/Investor-Emotion-and-Stock-Market-Behavior.git
cd Investor-Emotion-and-Stock-Market-Behavior
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Complete Pipeline

Run all analysis steps sequentially:

```bash
# Step 1: Extract emotions from social media messages
python src/step3_emotion_extraction.py

# Step 2: Aggregate emotions to firm-day level
python src/step4_aggregate_emotions.py

# Step 3: Run panel regression analysis
python src/step6A_panel_regression.py
```

### Individual Components

Each script can be run independently with custom parameters:

```bash
# Emotion extraction with custom model
python src/step3_emotion_extraction.py --input data/raw/messages.csv \
                                       --output data/processed/emotions.csv

# Aggregation with follower weighting
python src/step4_aggregate_emotions.py --input data/processed/emotions.csv \
                                       --output data/aggregated/firm_day.csv

# Regression with controls
python src/step6A_panel_regression.py --input data/aggregated/firm_day.csv \
                                      --output results/tables/
```

---

## 📊 Data Requirements

### Input Data Specifications

**Social Media Messages** (`data/raw/messages.csv`):
| Column | Description |
|--------|-------------|
| `message_id` | Unique message identifier |
| `firm_id` | Stock ticker or firm identifier |
| `user_id` | User identifier |
| `text` | Message content |
| `timestamp` | Message timestamp |
| `followers` | Number of followers (for weighting) |

**Stock Returns** (`data/raw/returns.csv`):
| Column | Description |
|--------|-------------|
| `firm_id` | Stock ticker |
| `date` | Trading date |
| `open_price` | Opening price |
| `close_price` | Closing price |
| `volume` | Trading volume |

---

## 🔑 Key Features

- **🤖 Transformer-based NLP**: State-of-the-art pretrained models for emotion classification
- **⚖️ Weighted Aggregation**: Accounts for differential user influence through follower counts
- **📈 Robust Econometrics**: Two-way fixed effects with clustered standard errors
- **🔄 Reproducible Pipeline**: Clear separation of processing, feature engineering, and analysis
- **📝 Publication-ready Output**: LaTeX-formatted regression tables

---

## 📈 Results

The analysis produces several regression specifications:

1. **Baseline Model**: Emotion signals → returns (with fixed effects)
2. **Full Model**: Adding financial controls (volume, volatility, lagged returns)
3. **Emotion Decomposition**: Individual emotion categories as separate predictors
4. **Robustness Checks**: Alternative aggregation methods and sample splits

All results are automatically saved in `results/tables/` in both CSV and LaTeX formats.

---

## 🛠️ Dependencies

**Core Libraries**:
- `pandas` >= 1.5.0 - Data manipulation
- `numpy` >= 1.23.0 - Numerical computing
- `torch` >= 2.0.0 - Deep learning framework
- `transformers` >= 4.30.0 - Pretrained NLP models
- `statsmodels` >= 0.14.0 - Econometric analysis
- `scikit-learn` >= 1.3.0 - Machine learning utilities

See `requirements.txt` for the complete dependency list.

---

## 📚 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@misc{investor_emotions_2024,
  author = {Bhanu Pratap},
  title = {Investor Emotions and Stock Market Behavior},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/bhanup23/Investor-Emotion-and-Stock-Market-Behavior}
}
```

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

Please ensure your code follows the existing style and includes appropriate documentation.

---



## 👤 Author

**Bhanu Pratap**

- GitHub: [@bhanup23](https://github.com/bhanup23)
- Repository: [Investor-Emotion-and-Stock-Market-Behavior](https://github.com/bhanup23/Investor-Emotion-and-Stock-Market-Behavior)

---

- Contact via GitHub profile

---

**⭐ If you find this project useful, please consider giving it a star!**
