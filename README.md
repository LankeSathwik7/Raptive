# Why Averages Always Look Normal

An interactive exploration of the Central Limit Theorem — one of the most important results in statistics.

## What This Is

A Streamlit dashboard that lets you pick any distribution — skewed, flat, bimodal — and watch what happens when you repeatedly sample from it and average the results. The averages always form a bell curve, no matter how odd the original data looks.

The app walks through the concept step by step, with interactive controls and plain-English explanations alongside the visualizations.


## Why the Central Limit Theorem

The CLT is the reason polls, quality control, A/B testing, and medical trials all work. It says that averages of random samples follow a normal distribution, regardless of how the underlying data is shaped — as long as samples are large enough.

This is a powerful idea that's worth seeing rather than just reading about. The app lets you experiment with it directly.


## Features

- Four parent distributions to choose from (Exponential, Uniform, Right-Skewed, Bimodal)
- Adjustable distribution shape, sample size, and number of samples
- Side-by-side comparison of raw data vs. sampling distribution
- Theoretical normal overlay to verify the CLT prediction
- Sample size progression showing how bigger samples produce tighter bell curves
- Key terms sidebar for readers who aren't statistics specialists
- Expandable math section for those who want the formal details


## Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open http://localhost:8501 in your browser.


## Built With

- Python
- Streamlit (interactive web framework)
- Plotly (charts)
- NumPy and SciPy (numerical computing and statistics)

---

Created as part of the Raptive Data Scientist assignment.
