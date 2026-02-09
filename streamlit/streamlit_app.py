"""
Why Averages Always Look Normal
A visual exploration of the Central Limit Theorem

"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# ── Page Setup ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Why Averages Always Look Normal", layout="wide")

PRIMARY = "#524CDE"
ACCENT = "#AD6FD8"

st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 1100px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.header("Settings")

distribution = st.sidebar.selectbox(
    "Pick a distribution",
    ["Exponential", "Uniform", "Right-Skewed", "Bimodal"],
)

DIST_CAPTIONS = {
    "Exponential": "Like wait times — most are short, a few are really long.",
    "Uniform": "Every value is equally likely, like rolling a fair die.",
    "Right-Skewed": "Values pile up on the left, trail off to the right — like incomes.",
    "Bimodal": "Two separate clusters — like mixing two very different groups.",
}
st.sidebar.caption(DIST_CAPTIONS[distribution])

# Shape parameters
st.sidebar.markdown("---")
st.sidebar.subheader("Shape")

if distribution == "Exponential":
    rate = st.sidebar.slider("Rate", 0.5, 3.0, 1.0, 0.1,
                             help="Higher = values packed closer to zero")
elif distribution == "Uniform":
    low = st.sidebar.slider("Minimum", 0.0, 5.0, 0.0, 0.5)
    high = st.sidebar.slider("Maximum", 6.0, 20.0, 10.0, 0.5)
elif distribution == "Right-Skewed":
    skew_amount = st.sidebar.slider("How skewed", 1.0, 5.0, 2.0, 0.5,
                                    help="Higher = longer right tail")
elif distribution == "Bimodal":
    separation = st.sidebar.slider("Gap between peaks", 2.0, 8.0, 5.0, 0.5)

# Sampling controls
st.sidebar.markdown("---")
st.sidebar.subheader("Sampling")

sample_size = st.sidebar.slider(
    "Sample size (n)", 2, 200, 30,
    help="How many values to grab each time before averaging")
num_samples = st.sidebar.slider(
    "Number of samples", 500, 5000, 2000, 500,
    help="How many times to repeat the grab-and-average process")

st.sidebar.button("Resample", help="Draw fresh random data with the same settings")

# Key terms
st.sidebar.markdown("---")
with st.sidebar.expander("Key terms explained"):
    st.markdown("""
**Distribution** — A picture of how values spread out. Tall bars = common values.

**Sample** — A handful of data points picked at random.

**Sample mean** — The average of one sample.

**Sampling distribution** — Take many samples, average each one, plot all those
averages. That's a sampling distribution.

**Central Limit Theorem** — No matter how weird your data looks, the averages
of random samples will form a bell curve — if the samples are big enough.

**Standard error** — How spread out the averages are.
Formula: *SE = σ / √n*. Bigger samples = smaller spread = tighter bell curve.
""")

# ── Generate Data ───────────────────────────────────────────────────────────

POP_SIZE = 50_000


def make_population():
    if distribution == "Exponential":
        return np.random.exponential(1 / rate, POP_SIZE)
    elif distribution == "Uniform":
        return np.random.uniform(low, high, POP_SIZE)
    elif distribution == "Right-Skewed":
        return np.random.beta(skew_amount, skew_amount * 4, POP_SIZE) * 10
    else:  # Bimodal
        half = POP_SIZE // 2
        left = np.random.normal(0, 1, half)
        right = np.random.normal(separation, 1, POP_SIZE - half)
        combined = np.concatenate([left, right])
        np.random.shuffle(combined)
        return combined


population = make_population()

# Vectorised sampling — fast even at high sample counts
indices = np.random.randint(0, len(population), size=(num_samples, sample_size))
sample_means = population[indices].mean(axis=1)

pop_mean = population.mean()
pop_std = population.std()
theoretical_se = pop_std / np.sqrt(sample_size)

# ── Helper for consistent chart styling ─────────────────────────────────────


def clean_layout(**overrides):
    base = dict(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        showlegend=False,
        margin=dict(t=10, b=40, l=50, r=20),
        height=340,
        xaxis=dict(gridcolor="#EEEEEE"),
        yaxis=dict(gridcolor="#EEEEEE"),
    )
    base.update(overrides)
    return base


# ── Main Content ────────────────────────────────────────────────────────────

st.title("Why Averages Always Look Normal")

st.markdown("""
Pick the weirdest-looking distribution you can find in the sidebar —
skewed, flat, two-humped, anything.
Then watch what happens when we take random samples and average them.
""")

st.divider()

# ── 1. The Raw Distribution ────────────────────────────────────────────────

st.subheader("Here's your distribution")

DIST_DESCRIPTIONS = {
    "Exponential": "Most values cluster near zero with a long tail to the right — "
                   "think of wait times at a busy restaurant.",
    "Uniform": "Completely flat. Every value between the minimum and maximum is "
               "equally likely.",
    "Right-Skewed": "Values bunch up on the left and stretch out to the right — "
                    "similar to how income is distributed.",
    "Bimodal": "Two distinct peaks — imagine mixing measurements from two very "
               "different groups.",
}
st.markdown(f"*{DIST_DESCRIPTIONS[distribution]}*")

fig_raw = go.Figure()
fig_raw.add_trace(go.Histogram(
    x=population, nbinsx=60, marker_color=ACCENT, opacity=0.85))
fig_raw.update_layout(**clean_layout(xaxis_title="Value", yaxis_title="Count"))
st.plotly_chart(fig_raw, use_container_width=True)

st.markdown("Clearly not a bell curve. Now here's where it gets interesting.")

st.divider()

# ── 2. The Central Limit Theorem in Action ─────────────────────────────────

st.subheader("Now watch what happens when we average")

st.markdown(f"""
We grabbed **{sample_size}** random values and averaged them.
Then repeated that **{num_samples:,}** times.
Here are all those averages, plotted right next to the original data:
""")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown(f"**Original data** *({distribution})*")
    fig_left = go.Figure()
    fig_left.add_trace(go.Histogram(
        x=population[:5000], nbinsx=50, marker_color=ACCENT, opacity=0.75))
    fig_left.update_layout(**clean_layout(
        height=320, xaxis_title="Value", yaxis_title="Count"))
    st.plotly_chart(fig_left, use_container_width=True)

with col_right:
    st.markdown("**Averages of samples** — *a bell curve appears*")
    fig_right = go.Figure()
    fig_right.add_trace(go.Histogram(
        x=sample_means, nbinsx=50, marker_color=PRIMARY,
        opacity=0.85, name="Sample means"))

    # Theoretical normal overlay
    x_curve = np.linspace(sample_means.min(), sample_means.max(), 200)
    y_curve = stats.norm.pdf(x_curve, pop_mean, theoretical_se)
    bin_width = (sample_means.max() - sample_means.min()) / 50
    y_curve = y_curve * num_samples * bin_width

    fig_right.add_trace(go.Scatter(
        x=x_curve, y=y_curve, mode="lines",
        name="Predicted bell curve",
        line=dict(color="#E74C3C", width=2.5)))
    fig_right.update_layout(**clean_layout(
        height=320, showlegend=True, xaxis_title="Sample mean",
        yaxis_title="Count",
        legend=dict(x=0.50, y=0.95, bgcolor="rgba(255,255,255,0.8)")))
    st.plotly_chart(fig_right, use_container_width=True)

st.markdown("""
The left chart can be any shape at all.
The right chart? Always a bell curve.
That's the **Central Limit Theorem** — one of the most important results
in all of statistics.
""")

st.divider()

# ── 3. Theory vs. Reality ──────────────────────────────────────────────────

st.subheader("Theory vs. reality")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Center (predicted)", f"{pop_mean:.3f}")
with c2:
    st.metric("Center (actual)", f"{np.mean(sample_means):.3f}",
              delta=f"{np.mean(sample_means) - pop_mean:+.4f}")
with c3:
    st.metric("Spread of averages",
              f"Actual: {np.std(sample_means):.4f}",
              delta=f"Predicted: {theoretical_se:.4f}",
              delta_color="off")

st.markdown(f"""
The CLT predicts averages should cluster around **{pop_mean:.3f}**
with a spread of **{theoretical_se:.4f}**.
Our results: center = **{np.mean(sample_means):.3f}**,
spread = **{np.std(sample_means):.4f}**.
Almost exact — and the match improves with more samples.
""")

st.divider()

# ── 4. Sample Size Effect ──────────────────────────────────────────────────

st.subheader("Bigger samples, tighter bell curve")

st.markdown("Watch the sampling distribution shrink as sample size grows:")

sizes = [5, 15, 50, 200]
colors = [ACCENT, "#7B68EE", PRIMARY, "#3A36B0"]

fig_grid = make_subplots(
    rows=1, cols=4,
    subplot_titles=[f"n = {s}" for s in sizes])

for i, n in enumerate(sizes):
    idx = np.random.randint(0, len(population), size=(1500, n))
    means = population[idx].mean(axis=1)
    fig_grid.add_trace(
        go.Histogram(x=means, nbinsx=30, marker_color=colors[i],
                     opacity=0.8, showlegend=False),
        row=1, col=i + 1)

fig_grid.update_layout(
    height=280, margin=dict(t=40, b=30, l=40, r=20),
    plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
st.plotly_chart(fig_grid, use_container_width=True)

st.markdown("""
At **n = 5** the averages are all over the place.
By **n = 200** they're tightly packed around the true mean.
This is exactly what the formula *SE = σ / √n* predicts:
quadruple the sample size, halve the spread.
""")

st.divider()

# ── 5. Why This Matters ────────────────────────────────────────────────────

st.subheader("Why this matters in the real world")

st.markdown("""
The Central Limit Theorem is the reason much of applied statistics works:

- **Polls and surveys** — You can estimate what millions think by asking
  a few thousand, because the sample average will be close to the true average.
- **Quality control** — A factory doesn't test every single item. A well-chosen
  sample is enough to estimate defect rates for the whole batch.
- **A/B testing** — When a website tests two button colors, the CLT is why they
  can declare a winner from a relatively small number of visitors.
- **Medical trials** — Researchers don't need to test a drug on everyone. A
  properly sized sample gives reliable results, thanks to this theorem.

The beauty of it: **you don't need to know what the original data looks like.**
As long as samples are large enough, the averages behave predictably — every time.
""")

# ── 6. The Math (for those who want it) ────────────────────────────────────

with st.expander("For the curious — the math behind it"):
    st.markdown(f"""
**Formal statement**

If X₁, X₂, …, Xₙ are independent and identically distributed with
mean μ and finite variance σ², then as n increases:

> **(X̄ − μ) / (σ / √n)  →  N(0, 1)**

The standardized sample mean converges to a standard normal distribution.

---

**Your current settings**

| Quantity | Value |
|---|---|
| Population mean (μ) | {pop_mean:.4f} |
| Population std dev (σ) | {pop_std:.4f} |
| Sample size (n) | {sample_size} |
| Theoretical SE = σ/√n | {theoretical_se:.4f} |
| Observed SE | {np.std(sample_means):.4f} |
| Difference | {abs(np.std(sample_means) - theoretical_se):.6f} |

---

**Rule of thumb:** n ≥ 30 is often cited as "big enough" for the CLT to produce
a good normal approximation. Highly skewed distributions may need larger n —
try Exponential with n = 5 vs n = 50 to see this yourself.
""")

# ── Footer ──────────────────────────────────────────────────────────────────

st.divider()
st.caption("Built by Sathwik | Raptive Data Scientist Assignment")
