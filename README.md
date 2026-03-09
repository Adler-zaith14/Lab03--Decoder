# LAB P1-03: Decoder do Transformer
**Disciplina:** Tópicos em Inteligência Artificial  
**Instituição:** ICEV  
**Autor:** Adler Castro Alves  

---

## Objetivo
Implementação from scratch do Decoder do Transformer conforme o paper "Attention Is All You Need" (Vaswani et al., 2017), utilizando apenas NumPy e Pandas. O laboratório cobre os três mecanismos centrais do Decoder: máscara causal (Look-Ahead Mask), cross-attention entre Encoder e Decoder, e o loop de inferência auto-regressiva.

```
Attention(Q, K, V) = softmax(QK^T / √d_k + M) V
```

---

## Como Executar
```bash
pip install -r requirements.txt
python decoder.py
```

---

## Estrutura
```
├── decoder.py          # Implementação principal
├── requirements.txt    # Dependências
└── README.md
```

---

## Máscara Causal
Durante o treinamento paralelo em GPU, a frase inteira entra no Decoder de uma vez. Para impedir que o token na posição `i` atenda à posição `i+1`, injetamos uma máscara triangular superior com `-inf` antes do softmax. O softmax transforma `-inf` em exatamente `0.0` — a posição some da atenção.

```python
def causal_mask(seq_len):
    m = np.full((seq_len, seq_len), -np.inf)
    return np.triu(m, k=1)
```

---

## Exemplo de Input e Output

```python
# Tarefa 1 — Máscara Causal
Q = np.random.randn(5, 64)
K = np.random.randn(5, 64)
scores = Q @ K.T + causal_mask(5)
probs = softmax(scores)

# Tarefa 2 — Cross-Attention
enc_out   = np.random.randn(1, 10, 512)
dec_state = np.random.randn(1,  4, 512)
out = cross_attention(enc_out, dec_state)

# Tarefa 3 — Loop Auto-Regressivo
ctx = ["<START>", "O", "rato"]
while True:
    probs = next_token(ctx, enc_out)
    token = idx2token[np.argmax(probs)]
    ctx.append(token)
    if token == "<EOS>" or len(ctx) >= max_tokens:
        break
```

**Inputs:**
```
Q, K:      (5, 64)        — scores da máscara causal
enc_out:   (1, 10, 512)   — saída do encoder
dec_state: (1,  4, 512)   — estado atual do decoder
```

**Outputs:**
```
probs:  (5, 5)   — linha i só enxerga posições <= i
out:    (1, 4, 512)
frase:  <START> O rato token_6 token_11 ... <EOS>
```

---

## Diferença entre Self-Attention e Cross-Attention

| | Self-Attention | Cross-Attention |
|---|---|---|
| Q | própria sequência | estado do Decoder |
| K, V | própria sequência | saída do Encoder |
| Máscara | sim (causal) | não |

Na cross-attention o Decoder tem permissão para olhar a frase original do Encoder por completo — sem restrição de posição.

**Anexo Google Colab:**
[https://colab.research.google.com/drive/1WAJ3ZHLuqVxDHQtS_pNq6OVOOkBWVuWH?usp=sharing]


**Referência:**  
* GOODFELLOW, Ian; BENGIO, Yoshua; COURVILLE, Aaron. Deep Learning. [S. l.]: MIT Press, 2016..
 * JURAFSKY, Daniel; MARTIN, James H. Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition with Language Models. 3. ed. draft. [S. l.]: Stanford University/University of Colorado at Boulder, 2026..
 * RASCHKA, Sebastian. Build a Large Language Model (From Scratch). 1. ed. [S. l.]: Manning (MEAP), 2021..
 * UNIVERSIDADE FEDERAL DO PIAUÍ. Estágio Curricular Supervisionado - Fábrica de Software I: normas para o estágio supervisionado. Teresina: UFPI, 2026..
 * VASWANI, Ashish et al. Atenção é tudo o que você precisa. Tradução de Machine Translated by Google. [S. l.]: Google Brain/Google Research, 2017..
