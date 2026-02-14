# 📊 Historical Pattern Matching by Nicola Chimenti

A **web-based pattern matching tool** built with **Streamlit** and **Python**, designed to find similar historical price patterns in financial markets using Dynamic Time Warping (DTW) and Pearson correlation.  
This tool helps traders and analysts identify historically similar market conditions and analyze their outcomes to make informed trading decisions.

**This is my Harvard CS50P Final Project**, Certificate: [View Certificate](https://certificates.cs50.io/1fbe53e1-1594-47fb-b311-aa7b3a91a6d6.pdf?size=letter)

#### Video Demo: **[Coming Soon]**

## 🌐 Live Demo

**Try it now:** [Historical Pattern Matching App](https://historycal-pattern-matching.streamlit.app/)  
No installation required — start analyzing market patterns immediately in your browser!

---

## 🧩 Features

### 📈 Multi-Algorithm Pattern Matching
Find similar historical patterns using two complementary algorithms:
- **Dynamic Time Warping (DTW)** for shape similarity
- **Pearson correlation** for directional similarity
- **Combined filtering** requiring both metrics to exceed thresholds
- **Sliding window approach** scanning all historical data
- **Percentage normalization** for scale-independent comparison

**Key Capabilities:**
- Yahoo Finance API integration for data retrieval
- Customizable similarity thresholds (DTW and correlation)
- Support for any ticker available on Yahoo Finance
- Real-time data validation and error handling
- Mobile-responsive interface

---

### 📊 Statistical Analysis
Performance metrics for matched patterns:
- **Return statistics** (average, median, best/worst case)
- **LONG/SHORT separation** for directional analysis
- **Maximum excursions** (positive and negative)
- **Win rate distribution** across matched patterns
- **Combined similarity score** ranking

**Analysis Features:**
- Separate statistics for bullish and bearish outcomes
- Excursion tracking to understand price movements
- Match details with dates and scores
- CSV export for further analysis in Excel

---

### 🎯 Flexible Configuration
Customize your analysis parameters:
- **Multiple timeframes**: 5m, 15m, 30m, 1h, 4h, 1d, 1wk, 1mo
- **Custom date ranges**: Set start date for historical data
- **Pattern end date**: Analyze patterns from specific dates
- **Pattern length**: 3-1000 periods (days, weeks, or candles)
- **Future horizon**: 1-1000 periods for outcome analysis
- **Dual thresholds**: Independent control of DTW and correlation filters

---

### 📉 Interactive Visualization
Multiple chart types for pattern comparison:
- **Current pattern display** with normalized percentage changes
- **Top 10 historical matches** overlay with color coding
- **Interactive Plotly charts** with zoom and hover details
- **Match labels** showing date and DTW similarity score
- **Separate subplots** for clear visual comparison

**Display Options:**
- Normalized to 0% starting point for direct comparison
- Color-coded historical patterns for easy differentiation
- Interactive legend for toggling patterns on/off
- Responsive design for mobile and desktop viewing

---

### 🌍 Multi-Language Support
Interface available in:
- 🇬🇧 **English** (default)
- 🇮🇹 **Italiano**
- 🇺🇦 **Українська**

Financial terminology (timeframe, LONG, SHORT, DTW, score) remains in English across all languages for consistency.

---

## 📦 Local Installation

If you prefer to run it locally or modify the code:

#### Prerequisites
- Python 3.8 or higher
- pip package manager

#### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/pattern-matching.git
cd pattern-matching
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run project.py
```

4. **Open your browser**  
The app will automatically open at `http://localhost:8501`

---

## 🚀 Usage

### Basic Analysis

1. **Configure parameters in sidebar**:
   - Enter ticker symbol (e.g., AAPL, MSFT, BTC-USD)
   - Select timeframe (1d recommended for daily analysis)
   - Optionally set custom start date for historical data
   
2. **Define pattern to analyze**:
   - Set pattern end date (default: today)
   - Choose period type (Days/Weeks/Candles)
   - Enter pattern length (e.g., 10 days)
   
3. **Set future horizon**:
   - Choose period type for analysis
   - Enter number of periods to analyze after pattern

4. **Adjust similarity criteria**:
   - DTW threshold: 0.0-1.0 (shape similarity)
   - Correlation threshold: -1.0-1.0 (direction similarity)

5. **Click "Run Analysis"**

6. **Review results**:
   - Aggregate statistics (total matches, returns, excursions)
   - LONG/SHORT case breakdown
   - Visual comparison charts
   - Detailed match table
   - CSV export option

---

## 🔧 Technical Details

**Built with:**
- **Streamlit**: Web framework for data applications
- **yfinance**: Yahoo Finance API wrapper for market data
- **FastDTW**: Fast Dynamic Time Warping implementation
- **SciPy**: Pearson correlation calculations
- **Plotly**: Interactive chart generation
- **Pandas & NumPy**: Data manipulation and numerical analysis
- **pytest**: Testing framework

**Algorithms:**
- **Dynamic Time Warping (DTW)**: Measures similarity between time series by finding optimal alignment
- **Pearson Correlation**: Measures linear relationship between return sequences
- **Sliding Window**: Iterates through historical data with fixed-length window
- **Percentage Normalization**: Converts prices to percentage changes from starting point

**Data Source:**
- Yahoo Finance API via yfinance library
- Historical data availability varies by asset
- Intraday and daily intervals supported

---

## 💼 Use Cases

This tool is designed for:
- **Quantitative Traders** identifying similar market conditions
- **Technical Analysts** finding historical pattern precedents
- **Algorithm Developers** backtesting pattern-based strategies
- **Market Researchers** studying price pattern repetition
- **Risk Managers** assessing potential outcomes based on history
- **Students** learning about time series analysis and pattern matching

---

## 🧪 Testing

Run the test suite:
```bash
pytest test_project.py -v
```

**Test Coverage:**
- Pattern similarity calculations with known inputs
- Pattern finding with synthetic repeating data
- Future performance analysis with trending data
- Statistical calculations with various scenarios
- Ticker validation for valid and invalid symbols
- Edge cases (empty data, strict thresholds, invalid tickers)

---

## ⚠️ Risk Disclaimer

**IMPORTANT**: This tool is for educational and research purposes only.

**Key Considerations:**
1. **Past patterns do not guarantee future outcomes**
2. **Market conditions evolve** - Historical similarities may not repeat
3. **Statistical significance** - Ensure adequate sample size (10+ matches recommended)
4. **Survivorship bias** - Data may not include delisted securities
5. **Data quality** - Yahoo Finance data should be verified for critical decisions
6. **Execution costs** - Real trading involves commissions, slippage, and spreads
7. **Market impact** - Large positions can affect prices

**Before trading:**
- Thoroughly understand the risks involved
- Never risk more than you can afford to lose
- Consider consulting with a financial advisor
- Implement proper risk management techniques
- Validate findings with out-of-sample testing

---

## 💬 Feedback & Contributions

If you find this tool useful or have ideas for improvement:
- Open an **[Issue](../../issues)** for bug reports or feature requests
- Submit a **Pull Request** to contribute enhancements
- Share your findings and use cases

---

## 📜 License

Distributed under the **MIT License** — free to use, modify, and share with proper attribution.

---

## 👤 Author

**Nicola Chimenti**  
Financial Data Analyst - Python & MQL4/5 Developer
Business Management Graduate  

🌐 [Personal Website](https://nicolachimenti.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/nicolachimenti/)  
💻 [GitHub](https://github.com/TeknoTrader)  
📧 Email: assistenza@nicolachimenti.com

---

## 🙏 Acknowledgments

This project was developed as my final project for **CS50P (CS50's Introduction to Programming with Python)** by Harvard University.

**Special Thanks:**
- Harvard CS50 team for instruction and support
- Yahoo Finance for providing market data API
- The Streamlit community for documentation
- Claude AI for assistance with translations

---

## ❓ FAQ

**Q: What makes this different from simple correlation?**  
A: DTW captures the shape of price movements, not just final outcomes. Two patterns with the same return can have very different paths. DTW + correlation together ensure both shape and direction similarity.

**Q: How many matches should I expect?**  
A: Depends on threshold settings. Lower thresholds (DTW 0.3, Correlation 0.2) find more matches but with less similarity. Higher thresholds (DTW 0.7, Correlation 0.6) find fewer but more similar patterns.

**Q: Can I use this for live trading?**  
A: This tool is designed for research and analysis. Live trading requires additional considerations like execution costs, slippage, and real-time risk management.

**Q: What's the minimum historical data needed?**  
A: At least 100+ periods beyond your pattern length for meaningful results. More data provides better statistical confidence.

**Q: Can I analyze cryptocurrencies?**  
A: Yes! Any asset with a Yahoo Finance ticker can be analyzed (e.g., BTC-USD, ETH-USD).

**Q: Why might I get zero matches?**  
A: Your thresholds may be too strict, or the current pattern may be unique. Try lowering DTW threshold to 0.4 and correlation to 0.2.

**Q: What does "Max Escursione Negativa" mean in LONG cases?**  
A: It shows how much the price dropped during the holding period, even if it ended positive. Important for understanding drawdown risk.

**Q: Can I export the results?**  
A: Yes, click the "Download CSV" button to export all match details with dates, scores, returns, and excursions.

---

## 📚 Methodology

### Pattern Matching Approach

The tool uses a two-stage filtering process:

1. **DTW Similarity (0-1 scale)**:
   - Measures shape similarity between normalized price patterns
   - Uses FastDTW algorithm with Euclidean distance
   - Exponential normalization: `similarity = exp(-distance/100)`
   - Higher values indicate more similar shapes

2. **Pearson Correlation (-1 to 1)**:
   - Measures directional similarity between return sequences
   - +1 = perfect positive correlation
   - 0 = no correlation
   - -1 = perfect negative correlation

3. **Combined Filtering**:
   - Pattern must satisfy BOTH thresholds (AND logic)
   - Combined score = (DTW + Correlation) / 2
   - Results sorted by combined score

### Statistical Metrics

**LONG Cases**: Final return > 0%  
**SHORT Cases**: Final return < 0%

**Max Positive Excursion**: Highest point reached during holding period  
**Max Negative Excursion**: Lowest point reached during holding period

These metrics show the full price range, not just entry and exit points.

---

⭐ **If you find this tool helpful, please give the repository a star!**

---

**VAT Code**: 02674000464  
**© 2026 Nicola Chimenti. All rights reserved.**
