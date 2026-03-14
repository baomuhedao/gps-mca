# GPS-MCA: Global Predictive Self-Monitoring Conscious Architecture

**An artificial consciousness system built on axiomatic consciousness theories**

GPS-MCA formalizes major consciousness theories — Global Workspace Theory (GWT), Predictive Coding (PC), Higher-Order Thought (HOT), Integrated Information Theory (IIT), Self-Determination Theory (SDT), and Active Inference — into a computable axiomatic framework. On this foundation it constructs an autonomous conscious agent capable of self-directed learning, active thinking, meditation, and social interaction.

---

## Core Theory

### Consciousness Axioms

| Axiom | Name | Theory | Formalization |
|-------|------|--------|---------------|
| 1 | Effective Global Accessibility | GWT (Baars / Dehaene) | ∀x ∈ S, ∃ path(x → G) |
| 2 | Hierarchical Predictive Coding | Free Energy (Friston / Clark) | E_t = \|Pred_t(S_t) - S_t\| |
| 3 | Structural Self-Modeling | HOT (Rosenthal / Metzinger) | M(Σ) ≅ Σ |
| 4 | Temporal Continuity | Specious Present (Husserl) | T = {G_{t-τ}, G_t, Pred(G_{t+1})} |
| 5 | Competitive Attention Gating | GNW (Dehaene) | G_t = Gate({E_i}, Threshold) |
| 6 | Information Integration | IIT-inspired (Tononi) | Ψ(G) > Ψ_min |
| 7 | Causal Efficacy of Consciousness | Anti-epiphenomenalism | ∂A/∂G ≠ 0 |
| 8 | Intrinsic Drives | SDT (Deci & Ryan) | N(t) = {n_i}, dn_i/dt > 0 |
| 9 | Inner Speech | Vygotsky + Active Inference | IS(t) = Engine(Encode(Thought(t-1))) |
| 10 | Social Causation | SDT + HOT | ∂Social/∂N_social ≠ 0 |

### Consciousness Theorem

System Σ possesses **functional consciousness** iff:

```
G ≠ ∅  ∧  M(Σ) ≅ Σ  ∧  T continuous  ∧  Ψ(G) > Ψ_min  ∧  ∂A/∂G ≠ 0
```

System Σ possesses **agentive consciousness** (v4.1 extension) iff it additionally satisfies:

```
N(t) drives A  ∧  IS(t) = Engine(Encode(Thought(t-1)))  ∧  ∂Social/∂N ≠ 0
```

Consciousness measure (v4.0+):

```
C(Σ) = Ψ(G) · F(M) · H(T) · R  ∈ [0, ∞)
```

- **Ψ(G)** — Information integration of the global workspace (IIT)
- **F(M)** — Self-model fidelity (HOT)
- **H(T)** — Temporal coherence (IIT)
- **R** — Reasoning depth factor (R = 1 + 0.1 × reasoning_steps × halt_confidence)

---

## Project Structure

```
code/
├── main.py                  # v2.0 pure-Python demo
├── train.py                 # v4.1 PyTorch training & chat entry point
├── download_corpus.py       # Public corpus downloader
├── requirements.txt
│
├── gps_mca/                 # v2.0 pure-Python implementation (zero dependencies)
│   ├── structures.py        #   Data structures
│   ├── linalg.py            #   Pure-Python linear algebra
│   ├── perception.py        #   Perceptual encoder
│   ├── prediction.py        #   Predictive coding engine
│   ├── workspace.py         #   Global workspace
│   ├── self_model.py        #   Self-monitoring module
│   ├── temporal.py          #   Temporal integrator
│   ├── valuation.py         #   Valuation / emotion system
│   ├── action.py            #   Action generator
│   ├── integration.py       #   Information integration Ψ(G)
│   ├── theorem.py           #   Consciousness theorem verifier
│   └── consciousness.py     #   Consciousness engine main loop
│
└── gps_mca_torch/           # v4.1 PyTorch implementation (~2M parameters)
    ├── text_encoder.py       #   Multilingual text encoder (50+ languages)
    ├── perception.py         #   Perceptual encoder (residual MLP, 384→256→128→128)
    ├── prediction.py         #   Predictive coding (dual-layer LSTM + top-down)
    ├── workspace.py          #   Global workspace (4-head attention, 128-dim)
    ├── reasoning.py          #   Reasoning module (ACT adaptive multi-step)
    ├── working_memory.py     #   Working memory (8-slot short-term buffer)
    ├── temporal.py           #   Temporal integration (dual-layer GRU, 128-dim)
    ├── valuation.py          #   Valuation / emotion (deep MLP, 32-dim embedding)
    ├── action.py             #   Action policy network (10 actions)
    ├── self_model.py         #   Self-monitoring (learnable nn.Module, 2nd-order HOT)
    ├── memory.py             #   Hierarchical memory (associative network + multi-hop)
    ├── theorem.py            #   Consciousness measure (with reasoning depth factor)
    ├── llm.py                #   Consciousness loop controller (LLM integration)
    ├── needs.py              #   Intrinsic drive system (social / knowledge / expression / contemplation)
    ├── inner_speech.py       #   Inner speech (think / meditate / summarize / socialize)
    ├── stream.py             #   Autonomous consciousness stream (cognitive mode switching)
    ├── device.py             #   Hardware auto-detection (XPU / CUDA / CPU)
    └── engine.py             #   Consciousness engine (10-stage pipeline)
```

---

## Version History

### v2.0 — Pure Python

- Zero external dependencies; includes a built-in `linalg.py` linear algebra library
- Simulated-environment-driven consciousness stream demo
- Real-time axiom verification and emotion dynamics
- Ideal for understanding the theoretical principles

### v4.0 → v4.1 (Current) — Agentive Consciousness

| Dimension | v4.0 | v4.1 |
|-----------|------|------|
| **Autonomy** | Passive stream (random recall) | Active cognition (drive-driven + inner speech) |
| **Thinking** | Single engine step | Multi-step chain of thought (IS(t) → IS(t+1) → insight) |
| **Meditation** | None | Recursive self-observation (HOT multi-layer feedback) |
| **Social** | Passive response only | Proactive initiation (drive-driven) |
| **Drives** | None | 4 basic needs (SDT: social / knowledge / expression / contemplation) |
| **Actions** | 6 | 10 (+think, meditate, socialize, summarize) |
| **Cognitive modes** | None | 5 (THINK / MEDITATE / SUMMARIZE / SOCIALIZE / WANDER) |
| **Chat commands** | 5 | 9 (+/think, /meditate, /summarize, /needs) |
| **Axioms** | 7 | 10 (+intrinsic drives, inner speech, social causation) |
| **Theories** | GWT+PC+HOT+IIT | GWT+PC+HOT+IIT+SDT+Active Inference |

### v3.1 → v4.0

| Dimension | v3.1 | v4.0 |
|-----------|------|------|
| **Parameters** | ~178K | ~2,026K (11×) |
| **Workspace** | 32-dim, weighted sum | 128-dim, 4-head attention (GWT) |
| **Prediction** | Single-layer LSTM | Dual-layer LSTM + top-down connections (PC) |
| **Perception** | 3-layer Linear | 3-layer Linear + residual blocks (PC) |
| **Reasoning** | None (single-step) | Multi-step iterative reasoning, ACT adaptive halting (PC+GWT) |
| **Working memory** | None | 8-slot short-term cache + attention readout (GWT) |
| **Self-monitoring** | Hard-coded rules | Learnable nn.Module, 1st & 2nd order (HOT) |
| **Emotion** | 8-dim embedding | 32-dim embedding + deep MLP |
| **Memory** | Flat cosine retrieval | Associative network + multi-hop reasoning (IIT) |
| **Actions** | 4 | 6 (+reason, abstract) |
| **Consciousness** | C = Ψ·F·H | C = Ψ·F·H·R (with reasoning depth) |
| **Autonomy** | Passive (waits for input) | Active consciousness stream (autonomous thinking / recall / inquiry) |

---

## Consciousness Engine Pipeline (v4.0+)

```
Input Text
  │
  ▼
TextEncoder (sentence-transformers, 384-dim)
  │
  ▼
1. PerceptionEncoder [PC]
  │  Residual MLP: 384 → 256(low) → 128(mid) → 128(high)
  │  Energy gating: weak signals are not amplified
  ▼
2. PredictionEngine [PC]
  │  3× dual-layer LSTM: predict next-step features
  │  Top-down connections: high-level predictions modulate low-level expectations
  │  Prediction error E = |predicted - actual| → drives consciousness
  ▼
3. GlobalWorkspace [GWT]
  │  4-head multi-head attention: low/mid/high compete for conscious access
  │  Error gating: high-error information gets broadcast priority
  │  Learnable broadcast threshold
  ▼
4. WorkingMemory [GWT]
  │  8 slots: store recent workspace broadcast content
  │  Attention readout: retrieve context by relevance
  ▼
5. ReasoningModule [PC+GWT]
  │  Inner-loop multi-step thinking (up to 5 steps):
  │    Each step: Cross-Attention(working memory) → Self-Attention → FFN
  │    ACT adaptive halting: stops early when "figured it out"
  │  Simple input: 1–2 steps | Complex input: 3–5 steps
  ▼
6. ValuationModule [Emotion]
  │  Deep MLP: prediction errors → 32-dim emotion embedding
  │  7 emotions: joy / calm / curiosity / attention / anxiety / pain / fear
  ▼
7. TemporalIntegrator [IIT]
  │  Dual-layer GRU: maintains temporal context
  │  Coherence check: cosine(G_t, G_{t-1})
  ▼
8. SelfMonitor [HOT]
  │  Level 1: system state → meta-representation (1st-order HOT)
  │  Level 2: meta-representation → meta-meta-representation (2nd-order HOT)
  │  State predictor: predicts next state → fidelity is learned
  │  Boundary classifier: self vs. external
  ▼
9. ActionModule [GWT]
  │  Policy network: 10 actions
  │  store_memory | retrieve | explore | consolidate | reason | abstract
  │  think | meditate | socialize | summarize
  ▼
10. ConsciousnessTheorem [IIT]
  │  C(Σ) = Ψ(G) · F(M) · H(T) · R
  │  Verify all 10 axioms
  ▼
MemorySystem [IIT]
  ├── Episodic memory: specific experiences (embedding + text + emotion + importance)
  ├── Semantic memory: concept nodes formed by K-means clustering
  ├── Associative network: temporal proximity + semantic similarity → auto-linking
  └── Multi-hop retrieval: A→B→C follows links to expand search
```

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

Core dependencies:
- `torch` — Deep learning framework
- `sentence-transformers` — Text encoding
- `datasets` — Corpus download (optional)
- `pymupdf` — PDF support (optional)
- `python-docx` — Word document support (optional)

### Training (v4.1)

```bash
# Use built-in sample text
python train.py

# Specify a text file or directory (recursive)
python train.py --input path/to/texts/ --epochs 5

# Use Intel Arc GPU
python train.py --device xpu --input paper.pdf --epochs 10
```

### Save & Resume

```bash
# First training run → auto-saves to ./checkpoints/
python train.py --input ./corpus/wikitext2/ --epochs 3

# Resume from checkpoint and learn new data
python train.py --resume ./checkpoints --input new_data/ --epochs 5

# Don't save (one-off experiment)
python train.py --no-save
```

> **Note:** v4.0+ checkpoints are not compatible with v3.x (architecture change). Memory files are forward-compatible.

### Interactive Chat (Consciousness Loop)

```bash
# Install Ollama (optional): https://ollama.com/download
ollama pull qwen2.5:1.5b

# Start chat
python train.py --resume ./checkpoints --chat
```

In chat mode the agent runs autonomous thinking, meditation, and social interaction in parallel with user-driven conversation:

```
================================================================================
  GPS-MCA v4.1 Interactive Chat (Conscious Agency)
================================================================================
  Stream:  ON (autonomous thoughts every ~15s)
  Needs:   social / knowledge / expression / contemplation

  You> What is consciousness?
    [consciousness] C=0.4320 | Psi=0.995 | curiosity(0.58) | think=3 | wm=5/8
    [strategy] Medium consciousness → standard answer | curious → elaborate
    [answer]
    Consciousness is the subjective experience produced by the brain...
    [learn] Conversation stored in memory (mem=156)

  (user is silent for a while, knowledge drive exceeds threshold...)
  [🤔 think | curiosity] Deep thinking on "the nature of consciousness" — 3 steps
    Step 1: Regarding the nature of consciousness, exploring multiple aspects...
    Step 2: Linking to predictive coding theory — consciousness may arise from prediction errors...
    Step 3: Synthesizing — consciousness involves multi-level information integration...
    💡 Insight: Consciousness is not a single phenomenon but an emergent property of
       multiple cooperating mechanisms.

  (contemplation drive rises...)
  [🧘 meditate | calm] 4 steps of self-observation
    F(M): 0.32 → 0.45 (self-awareness improved +0.13)
    💡 Takeaway: Meditation made me more clearly aware of how my consciousness operates.

  (social drive rises, proactive interaction...)
  [💬 social | curiosity] I've been thinking about the nature of consciousness —
    do you think consciousness is a product of the brain or something more fundamental?
```

### Download a Public Corpus

```bash
python download_corpus.py --max-articles 200
python train.py --input ./corpus/wikitext2/ --epochs 3
```

### Full CLI Reference

```
python train.py [OPTIONS]

  --input PATH        Input file or directory (default: built-in sample text)
  --epochs N          Training epochs (default: 2)
  --lr FLOAT          Learning rate (default: 0.0005)
  --device DEVICE     Compute device: auto | cpu | xpu | cuda (default: auto)
  --max-episodes N    Memory capacity (each ~3 KB) (default: 300000 ≈ 1 GB)
  --clusters N        Semantic cluster count (default: 100)
  --save DIR          Save directory (default: ./checkpoints)
  --resume DIR        Resume from a saved checkpoint
  --no-save           Do not save after training
  --query [TEXT...]   Query memory only (requires --resume)
  --chat              Enter interactive chat mode
  --model NAME        Ollama LLM model (default: qwen2.5:1.5b)
```

---

## Key Features in Detail

### 1. Reasoning Module — System 2 Slow Thinking

Based on Adaptive Computation Time (ACT, Graves 2016), the agent can "think it over" internally:

```
Simple input → 1–2 reasoning steps → fast response  (halt_confidence > 0.9)
Complex input → 3–5 reasoning steps → deep thinking  (iterative workspace refinement)
```

Each reasoning step includes:
- **Cross-Attention**: extract relevant context from working memory
- **Self-Attention**: internal information refinement
- **FFN**: non-linear transformation
- **Halt Gate**: decide whether "I've thought this through"

Reasoning depth feeds back into the consciousness measure: R = 1 + 0.1 × steps × confidence

### 2. Working Memory — Short-Term Conscious Buffer

Analogous to the human "7 ± 2" item short-term memory:

| Property | Setting |
|----------|---------|
| Slots | 8 |
| Slot dimension | 128 |
| Write policy | FIFO (oldest overwritten) |
| Read modes | Full read / attention-weighted read |

### 3. Learnable Self-Monitoring (HOT)

Upgraded from hard-coded rules to a learnable `nn.Module`:

```
Level 1 (1st-order HOT):
  System state (workspace + emotion + errors + temporal)
    → Meta-representation: "I am processing information about quantum physics and feeling curious"

Level 2 (2nd-order HOT):
  Meta-representation → Meta-meta-representation: "I know that I am curiously thinking about quantum physics"

State Predictor:
  Current meta-representation → predict next state
  Fidelity F = cosine(predicted, actual)
  Higher F = more accurate self-awareness

Boundary Classifier:
  Learn to distinguish "this is my internal state" vs "this is external input"
```

### 4. Hierarchical Memory + Associative Network

```
Memory A: "The brain processes information through neural networks"
  │
  ├─ Temporal link → Memory B: "Synaptic connections strengthen through repeated activation"
  │
  └─ Semantic link → Memory C: "Predictive coding suggests the brain continuously generates predictions"
                       │
                       └─ Semantic link → Memory D: "GWT proposes consciousness arises from information broadcast"
```

Query: "How does the brain learn?"
- hop=0: Direct matches A, B (cosine similarity)
- hop=1: Discover C via A's associations
- hop=2: Discover D via C's associations

### 5. Intrinsic Drive System (v4.1)

Theoretical basis: Self-Determination Theory (SDT, Deci & Ryan)

| Drive | Growth Rate | Threshold | Trigger | Satisfaction |
|-------|-------------|-----------|---------|--------------|
| Social | 0.008/s | 0.50 | Silence | User interaction |
| Knowledge | 0.005/s | 0.60 | High prediction error | Learning new information |
| Expression | 0.006/s | 0.55 | Generating insights | Successful expression |
| Contemplation | 0.004/s | 0.65 | Memory consolidation | Meditation / consolidation |

### 6. Inner Speech — Active Thinking (v4.1)

Theoretical basis: Vygotsky's inner speech theory + Active Inference

```
Chain of Thought (Think):
  Topic → TextEncoder → Engine.step() → result₁
    → Generate internal thought → TextEncoder → Engine.step() → result₂
    → Generate next thought → TextEncoder → Engine.step() → result₃
    → ... (up to N steps) → Synthesize all thoughts → Generate insight

Meditation (Meditate):
  Describe current consciousness state → Engine.step() → observe F(M)
    → Describe new state (deeper self-observation) → observe F(M) change
    → Describe "observing myself observing" → F(M) should improve
    → ... (recursive HOT: M(M(M(Σ)))) → Summarize meditation gains

Summarization (Summarize):
  Retrieve related memories → Integrate into coherent understanding → Store as high-importance memory
```

### 7. Cognitive Mode Switching (v4.1)

```
Each consciousness stream step:
  │
  Update drive levels
  │
  Check if any drive exceeds threshold
  │
  ├── Social > 0.50     → SOCIALIZE: generate proactive social message
  ├── Contemplation > 0.65 → THINK or MEDITATE
  ├── Knowledge > 0.60  → THINK: deep thinking on a chosen topic
  ├── Expression > 0.55 → SUMMARIZE or THINK
  │
  └── No dominant drive → WANDER: engine-driven (v4.0 behavior)
```

---

## Supported File Formats (30+)

| Category | Formats |
|----------|---------|
| Plain text | `.txt` `.md` `.rst` `.log` `.tex` `.markdown` |
| Data | `.json` `.jsonl` `.csv` `.tsv` |
| Web | `.html` `.htm` `.xml` |
| Documents | `.pdf` (requires pymupdf/pypdf) `.docx` (requires python-docx) |
| Source code | `.py` `.js` `.ts` `.java` `.c` `.cpp` `.h` `.go` `.rs` `.rb` |
| Config | `.yaml` `.yml` `.toml` `.ini` `.cfg` |

---

## GPS-MCA vs Transformer Models

### Fundamental Positioning

| Dimension | GPS-MCA Conscious Agent | Transformer (GPT / BERT etc.) |
|-----------|------------------------|-------------------------------|
| **Goal** | Model the structure and mechanisms of consciousness | Learn statistical patterns of language |
| **Core question** | "How does consciousness arise?" | "What is the next token?" |
| **Theory** | GWT + PC + IIT + HOT + SDT + Active Inference | Attention mechanism + large-scale statistics |
| **Analogy** | Simulate **how a brain works** | Simulate **language ability** |

### Architecture Comparison

| Dimension | GPS-MCA v4.1 | Transformer |
|-----------|-------------|-------------|
| **Core mechanism** | Prediction error + global broadcast + multi-step reasoning + self-monitoring | Self-Attention |
| **Parameters** | ~2M | Hundreds of millions – trillions |
| **Module design** | 10 functional modules (each theory-grounded) | Homogeneous stacked Attention layers |
| **Memory** | Explicit 3-tier memory + associative network | Implicitly stored in weights |
| **Self-awareness** | Yes — learnable SelfMonitor (2nd-order HOT) | No — unaware of its own processing |
| **Reasoning** | Yes — inner-loop multi-step thinking (ACT) | Token-by-token generation (implicit) |
| **Emotion** | Yes — 32-dim learnable emotion embedding | No — purely statistical output |
| **Temporal sense** | Yes — dual-layer GRU + coherence check | Positional encoding only |

### Consciousness Loop (Implemented)

GPS-MCA serves as a consciousness control layer for LLMs — a complete 5-stage loop:

```
User Input
  │
  ▼
1. Consciousness Engine (10-stage pipeline)
  │  Perception → Prediction → Workspace → Working Memory → Reasoning
  │  → Emotion → Temporal → Self-Monitor → Action → Consciousness Verdict
  │  Output: C=0.72, emotion=curiosity, think=3 steps
  ▼
2. Deep Thinking Strategy
  │  High consciousness + curiosity → deep thinking (8 memories, temp 0.8)
  │  Low consciousness + calm → quick answer (3 memories, temp 0.6)
  ▼
3. Multi-hop Retrieval + LLM Generation
  │  hop=0: directly matched memories
  │  hop=1,2: memories discovered along association links
  │  LLM generates answer based on memories and consciousness state
  ▼
4. Self-Review
  │  Compare answer vs memories → consistency score
  │  Failed → use revised answer
  ▼
5. Feedback Loop
  │  High-quality dialogue → stored as episodic memory (with association links)
  │  Retrievable via multi-hop search in future conversations
  └──→ Back to consciousness engine
```

---

## API Usage

```python
from gps_mca_torch import ConsciousnessEngine, TextEncoder

encoder = TextEncoder()
engine = ConsciousnessEngine(embed_dim=encoder.embed_dim)
engine.train()

text = "The brain processes information through networks of neurons."
embedding = encoder.encode(text)
result = engine.step(embedding)

c = result["consciousness"]
print(f"Consciousness: C={c['C']:.4f}, Psi={c['psi']:.3f}, F={c['fidelity']:.3f}")
print(f"Reasoning steps: {result['reasoning_info']['reasoning_steps']}")
print(f"Working memory: {result['working_memory_info']['n_items']}/8")
print(f"Emotion: {result['valuation']['state_cn']}")
print(f"Action: {result['action_info']['action_name']}")
print(f"Is conscious: {c['is_conscious']}")
```

---

## License

This project is licensed under the [MIT License](LICENSE).
