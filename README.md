# LAB P1-03: Implementando o Decoder
**Disciplina:** Tópicos em Inteligência Artificial – 2026.1
**Professor:** Prof. Dimmy Magalhães
**Instituição:** iCEV - Instituto de Ensino Superior
**Autor:** Adler Castro Alves

---

## Objetivo
Implementação from scratch dos blocos matemáticos centrais do Decoder do Transformer, utilizando apenas NumPy e Pandas. O lab cobre mascaramento causal, cross-attention entre Encoder e Decoder, e o loop de inferência auto-regressiva.

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

## O que foi implementado

**Tarefa 1 — Máscara Causal (Look-Ahead Mask)**

Função `causal_mask(seq_len)` que retorna uma matriz quadrada com `-inf` no triângulo superior. Quando somada aos scores antes do softmax, as posições futuras viram exatamente `0.0` — o token na posição `i` só enxerga até `i`.

Prova real: multipliquei Q e K fictícios, apliquei a máscara e o softmax, e imprimi a matriz de probabilidades confirmando que tudo acima da diagonal zerou.

**Tarefa 2 — Cross-Attention**

Função `cross_attention(enc_out, dec_state)` onde:
- Q vem do `decoder_state` — o que já foi gerado
- K e V vêm do `encoder_output` — a frase original

O Decoder pode olhar a frase do Encoder por completo, sem máscara. Tensores simulados: `enc_out (1, 10, 512)` e `dec_state (1, 4, 512)`.

**Tarefa 3 — Loop Auto-Regressivo**

Função mock `next_token(ctx, enc_memory)` que devolve um vetor de probabilidades sobre o vocabulário. O loop while chama ela a cada passo, aplica `argmax`, faz append do token escolhido na lista de contexto e para quando sai `<EOS>` ou atinge o limite.

```
<START> O rato token_6 token_11 token_7 token_13 token_2 token_13 token_5 token_16 <EOS>
```

---


**Anexo Google Colab:**
[https://colab.research.google.com/drive/1WAJ3ZHLuqVxDHQtS_pNq6OVOOkBWVuWH?usp=sharing]


**Referência:**  
* GOODFELLOW, Ian; BENGIO, Yoshua; COURVILLE, Aaron. Deep Learning. [S. l.]: MIT Press, 2016..
 * JURAFSKY, Daniel; MARTIN, James H. Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition with Language Models. 3. ed. draft. [S. l.]: Stanford University/University of Colorado at Boulder, 2026..
 * RASCHKA, Sebastian. Build a Large Language Model (From Scratch). 1. ed. [S. l.]: Manning (MEAP), 2021..
 * UNIVERSIDADE FEDERAL DO PIAUÍ. Estágio Curricular Supervisionado - Fábrica de Software I: normas para o estágio supervisionado. Teresina: UFPI, 2026..
 * VASWANI, Ashish et al. Atenção é tudo o que você precisa. Tradução de Machine Translated by Google. [S. l.]: Google Brain/Google Research, 2017..
