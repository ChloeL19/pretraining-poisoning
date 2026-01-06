# Plotting Best Practices for Academic Papers

A collection of tips and best practices for creating publication-quality figures.

---

## Error Bars: Use the BCa Method

*From Scott Emmons*

Use the **bias-corrected and accelerated ("BCa")** method for confidence intervals. SciPy's `bootstrap` function uses BCa as the default.

### Why BCa over Percentile?

- Seaborn uses the percentile method by default (called "ci")
- BCa is generally more accurate
- Main drawback: more complex to implement (but SciPy handles this)

### Example: Using BCa in Seaborn

```python
from scipy.stats import bootstrap
import seaborn as sns
import numpy as np

def bca_errorbar(data, confidence_level=0.95):
    """Custom error bar function using BCa bootstrap method."""
    data = np.asarray(data)
    # Remove NaN values
    data = data[~np.isnan(data)]
    if len(data) < 2:
        return (np.nan, np.nan)

    res = bootstrap(
        (data,),
        np.mean,
        confidence_level=confidence_level,
        method='BCa'
    )
    return (res.confidence_interval.low, res.confidence_interval.high)

# Use in seaborn:
sns.lineplot(data=df, x='x', y='y', errorbar=bca_errorbar)
```

---

## Seaborn Setup for Papers

*From Sam Toyer*

### Recommended Setup

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Set the context to "paper"
sns.set_context("paper")

# Enable dark grid
sns.set_style("darkgrid")

# Enable constrained layout (prevents cropping issues)
plt.rcParams['figure.constrained_layout.use'] = True
```

### Key Points

- Use seaborn for plotting where possible (e.g., `sns.lineplot` over `plt.plot`)
- Export as **PDF** for vector graphics
- Set figure dimensions to match paper layout (in inches) so LaTeX doesn't resize

### Common Figure Widths

For ICML 2023 style:
- `\textwidth` = 6.75 inches (full width)
- `\columnwidth` = 3.25 inches (single column)

---

## Making Matplotlib Paper-Ready

*From Adam Gleave*

### Step 1: Determine Paper Format

Find from the conference `.tex` example:
- Text width (e.g., 5.5 inches)
- Font family (e.g., Times New Roman)
- Font weight (e.g., 10pt)

**How to find text width:**
```latex
% In preamble:
\usepackage{layout}
% After \begin{document}:
\layout
```

**How to find font:**
```latex
\showthe\font
```

### Step 2: Create Style Sheets

Separate logic from style using matplotlib style sheets:
- One shared stylesheet for document-wide settings (font family, weight)
- Separate stylesheets for figure-specific settings (1-col vs 2-col)

```python
# Apply stylesheet via context manager
with plt.style.context(style_dict):
    # plotting code here
    pass
```

### Step 3: Handle Whitespace Properly

**DO NOT use `tight_layout`** - it crops/expands images, making font sizes inconsistent.

Better options:
1. **Easy**: Use `constrained_layout` (constraint solver approach)
   ```python
   plt.rcParams['figure.constrained_layout.use'] = True
   ```
2. **Advanced**: Manually tune `figure.subplot.{left,right,top,bottom}`

### Step 4: Save as Vector Graphics

- **PDF** is easiest for LaTeX
- SVG also works
- For maximum consistency, use LaTeX backend:
  ```python
  plt.rcParams['text.usetex'] = True
  ```
  (Not necessary if no math or LaTeX special formatting needed)

---

## Quick Reference Template

```python
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import bootstrap
import numpy as np

# --- Setup ---
sns.set_context("paper")
sns.set_style("darkgrid")
plt.rcParams['figure.constrained_layout.use'] = True

# --- Figure dimensions for ICML ---
TEXTWIDTH = 6.75  # inches
COLUMNWIDTH = 3.25  # inches

# --- BCa bootstrap error bars ---
def bca_errorbar(data, confidence_level=0.95):
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    if len(data) < 2:
        return (np.nan, np.nan)
    res = bootstrap((data,), np.mean, confidence_level=confidence_level, method='BCa')
    return (res.confidence_interval.low, res.confidence_interval.high)

# --- Create figure at correct size ---
fig, ax = plt.subplots(figsize=(COLUMNWIDTH, COLUMNWIDTH * 0.75))

# --- Plot with BCa error bars ---
sns.lineplot(data=df, x='x', y='y', errorbar=bca_errorbar, ax=ax)

# --- Save as PDF ---
fig.savefig('figure.pdf', format='pdf')
```

---

## Additional Tips

1. **Make plots self-explanatory**: Add text, labels, arrows to help readers without context
2. **Consistent sizing**: Generate figures at final size to avoid font scaling issues in LaTeX
3. **Use vector graphics**: PDF for LaTeX, SVG for web
4. **Version control plotting code**: Commit code that produces exact plots (per project CLAUDE.md)
