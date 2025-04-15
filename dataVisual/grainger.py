from tabulate import tabulate

# Data
lags = [1, 2, 3, 4, 5, 6, 7]
p_values = [0.8992, 0.6495, 0.2099, 0.3618, 0.5135, 0.4127, 0.4007]

# Prepare table content
table = [["Lags", "p-value (F-test)"]] + [[lag, f"{p:.4f}"] for lag, p in zip(lags, p_values)]

# Generate LaTeX table using tabulate
latex_table = tabulate(table[1:], headers=table[0], tablefmt="latex")

# Wrap in a full LaTeX document
latex_doc = r"""\documentclass{article}
\usepackage{booktabs}
\begin{document}

\section*{Granger Causality Test Results}

""" + latex_table + "\n\n\\end{document}"

# Write to .tex file
with open("granger_results.tex", "w") as f:
    f.write(latex_doc)

print("LaTeX file 'granger_results.tex' created.")