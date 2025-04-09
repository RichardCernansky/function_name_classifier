import re
import math

C_OPERATORS = {"+", "-", "*", "/", "%", "=", "==", "!=", ">", "<", ">=", "<=", "&&", "||", "!", "&", "|", "^", "<<", ">>",
               "++", "--", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="}

C_KEYWORDS = {"if", "else", "while", "for", "return", "switch", "case", "break", "continue", "struct", "typedef",
              "sizeof", "do", "goto", "default", "enum", "union", "static", "volatile", "extern", "const", "inline"}

def compute_halstead(c_code):
    """ Compute Halstead complexity metrics for C code. """
    

    tokens = re.findall(r'\w+|[^\s\w]', c_code)


    operators = [token for token in tokens if token in C_OPERATORS or token in C_KEYWORDS]
    operands = [token for token in tokens if token.isidentifier() and token not in C_KEYWORDS]

    n1 = len(set(operators))  
    n2 = len(set(operands))  
    N1 = len(operators)    
    N2 = len(operands)    

    # Compute Halstead metrics
    n = n1 + n2  # Vocabulary size
    N = N1 + N2  # Program length

    if n == 0:  # Avoid log(0)
        return {"volume": 0, "difficulty": 0, "effort": 0}

    V = N * math.log2(n)  # Halstead Volume
    D = (n1 / 2) * (N2 / n2) if n2 != 0 else 0  # Halstead Difficulty
    E = D * V  # Halstead Effort

    return {
        "volume": round(V, 4),
        "difficulty": round(D, 4),
        "effort": round(E, 4)
    }