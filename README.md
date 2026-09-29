# hex study
Hexagonal Geometry
Levant Ai
www.linkedin.com/in/khaled-alharrawi-908244431

# Does Hexagonal Geometry Help a Graph Neural Network? An Ablation Study

**Author / affiliation:** [Your name] · [Company name]
**Status:** small-scale ablation study. It confirms a known principle in a new setting; it is not a new discovery.

## The short version

We asked whether a GNN with a fixed hexagonal neighborhood beats one with random wiring of the same degree, and if so, why. After a first result that looked decisive turned out to be an under-training artifact, we rebuilt the experiment with a real convergence protocol and four controlled conditions.

| Condition | Test accuracy | Gap to hex_full |
|---|---|---|
| hex_full (hexagonal, direction-consistent weights) | 94.58% | – |
| hex_perm_full (same edges, direction consistency destroyed) | 84.21% | 10.36 pts |
| local_random_full (random, spatially local edges) | 81.44% ± 3.89 | 13.14 pts |
| global_random_full (random, no spatial constraint) | 83.21% ± 1.82 | 11.37 pts |

The three non-hex conditions land close together despite very different clustering and locality. The gain comes from binding each weight matrix to a fixed directional slot (the same principle as CNN kernels), not from hexagonal shape, locality, or clustering.

## What this is not

- Not a new discovery. Position-specific weights are the reason CNNs work; we showed the effect carries over to an irregular graph setting.
- Not evidence that graph models are the right tool for MNIST. We did not run CNN or MLP baselines, and a CNN would very likely match or beat these results on a regular pixel grid.

## Setup

- Data: MNIST downsampled to 14×14 with 2×2 average pooling (196 nodes). 50k train / 10k validation / 10k test.
- Model: GCNII-style GNN, 12 layers, hidden size 16, LayerNorm + GELU, Top-3-of-6 gating with a straight-through estimator, one weight matrix per neighbor slot.
- Training: Adam, lr 1e-3, batch 128, gradient clipping 1.0, early stopping on validation accuracy (patience 6, max 60 epochs). Test accuracy is measured once, at the best-validation checkpoint.
- Seeds: 5 model seeds for the hex conditions. For the random conditions, 8 random topologies × 3 model seeds, averaged per topology first.
- All graphs are 6-regular with 588 edges. The hex graph wraps around as a torus so there are no boundary nodes.
- Locality control: random 6-regular graphs whose edges all lie within torus (Chebyshev) distance 3.

## Limitations

- **Statistical power.** With 8 random topologies the smallest possible permutation p-value is 0.1111 (1/9). Hex beat every random topology (16 of 16), but formal significance (p < 0.05) would need at least 19.
- **Small scale.** 196 nodes on downsampled MNIST.
- **Missing baselines.** No CNN, MLP, square-grid, or Watts–Strogatz controls.
- **Architecture sensitivity.** A separate reimplementation with a different gate and a larger hidden size did not reproduce the same gap. The effect may shrink with capacity; we did not test this.
- **A data disclosure.** A power cut interrupted the run after `hex_full` had finished. Its five per-seed results were transcribed by hand into `results_raw.json` from the console output of an earlier, identical run (a later rerun reproduced the same numbers exactly for the seeds it covered, so the setup is deterministic on our hardware). Those five entries therefore have empty learning curves. All other conditions were saved directly by the script.

## Reproduce

```bash
pip install torch torchvision networkx numpy
python topology_check.py            # structural sanity check of all graphs, no training
python hex_locality_convergence.py  # full experiment, resumes from results_raw.json if present
```

Environment used: Windows, NVIDIA RTX 3070, Python 3.12, PyTorch 2.6.0 (CUDA 12.4). The full run takes roughly half a day on that GPU.

## Files

- `hex_locality_convergence.py`: main experiment and analysis
- `topology_check.py`: graph construction checks
- `results_raw.json`: raw per-run results
- `report_en.md`, `report_ar.md`: the full story of how we got here

## License

MIT (add a LICENSE file).

---
النتيجة: hex_full يصل إلى 94.58%، بينما الشروط الثلاثة الأخرى (hex_perm وlocal_random وglobal_random) متقاربة عند 81–84%. أي أن السبب ليس الشكل السداسي ولا المحلية ولا العنقدة، بل ربط كل مصفوفة وزن باتجاه ثابت، وهو مبدأ الشبكات الالتفافية المعروف.

هذه دراسة تؤكد مبدأً معروفاً في سياق جديد، وليست اكتشافاً جديداً. القيود مذكورة أعلاه، ومنها غياب CNN وMLP كخطوط أساس، ومحدودية القوة الإحصائية (p = 0.1111 عند 8 توبولوجيات)، وأن نتائج hex_full الخمس أُدخلت يدوياً في `results_raw.json` بعد انقطاع كهرباء (بلا منحنيات تعلم).
