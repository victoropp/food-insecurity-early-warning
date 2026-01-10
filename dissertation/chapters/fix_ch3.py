import re

with open('03_methods.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix all the corrupted patterns
fixes = [
    (r'^egin{', r'\begin{'),
    (r'^nd{', r'\end{'),
    (r' egin{', r' \begin{'),
    (r' nd{', r' \end{'),
    (r'\text{', r'\text{'),
    (r'\textbf{', r'\textbf{'),
    (r'\textit{', r'\textit{'),
    (r'\texttt{', r'\texttt{'),
    (r'\tcitep{', r'\citep{'),
    (r'\tcitet{', r'\citet{'),
    (r'\turl{', r'\url{'),
    (r'\titem ', r'\item '),
    (r' geq ', r' \geq '),
    (r' leq ', r' \leq '),
    (r' quad ', r' \quad '),
    (r'\$geq\$', r'$\geq$'),
    (r'\$leq\$', r'$\leq$'),
    (r'^subsection{', r'\subsection{'),
    (r'^section{', r'\section{'),
    (r'Thissimbalanced', r'This imbalanced'),
    (r'simportance', r's importance'),
]

for pattern, replacement in fixes:
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

with open('03_methods.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed")
