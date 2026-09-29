# The Hexagon Experiment: From a "Striking Discovery" to a Modest, Honest Result

This experiment started with a simple question: does a graph neural network (GNN) with a fixed hexagonal neighborhood (six neighbors per node, inspired by a honeycomb) really beat a network with the same number of connections wired at random? And if it does, why? Is it the hexagonal shape itself, or something more general?

The answer we ended up with, after weeks of testing, fixing, and doubting our own numbers, is quite different from where we started. I think the way we got there is worth documenting as much as the result itself.

## The start: a result that looked decisive

The first experiment trained a GCNII-style GNN on downsampled MNIST (14×14 pixels, 196 nodes). We compared a hexagonal graph, where every node has six neighbors in a fixed directional order (the first slot always means the same compass direction), against a random regular graph with the same degree. We also built a "kill test": the exact same neighbors as the hex graph, but with the slot order shuffled randomly for every node.

The first run showed the consistent hex graph beating the shuffled version by 25.76 percentage points. That looked like clear proof that the benefit was geometric and not just extra parameters. It was striking enough that we started thinking about publishing it.

## First doubt: the training budget was too small

Before publishing, we ran one simple check: the same code, changing only the number of training epochs (from 3 to 15). The result was uncomfortable. The "decisive" gap of 25.76 points shrank to 10.14 once the models had actually converged, and one of the secondary comparisons even flipped sign between the two budgets.

What we had taken for a finding was mostly an under-training artifact. Two networks with the same number of parameters converge at different speeds, and after only three epochs one of them had simply pulled ahead by luck of initialization.

The first lesson: a number from a run that hasn't converged isn't a measurement. It's a snapshot from the middle of a race.

## Second doubt: three things were changing at once

A closer review exposed a bigger problem. The hexagonal and random graphs don't differ in one property; they differ in three at the same time:

- Clustering coefficient (0.40 versus 0.018).
- Slot consistency (in the hex graph, slot k always means the same direction; in the random graph it means nothing).
- Spatial locality (hex neighbors are truly adjacent on the image; a random regular graph can link one corner of the image to the opposite corner).

Any gap between the two could come from any of these three, or from a mix, and we had no way to separate them.

## Building the deciding experiment

To answer properly we built four comparisons instead of two, plus a much stricter training protocol: a separate validation set, real early stopping instead of a fixed epoch count, and incremental saving of results (after a power cut cost us an entire run, which taught us that lesson the hard way). Along the way, testing on real hardware exposed two genuine bugs in our own code, including a graph-construction routine that worked by luck on one seed and failed on another.

The four conditions:

- **hex_full**: the original hexagonal graph with all three properties together.
- **hex_perm_full**: exactly the same edges as hex (same clustering, same locality), but with the directional consistency deliberately destroyed.
- **local_random_full**: a random graph whose edges are all spatially local, which separates locality from clustering and consistency.
- **global_random_full**: a fully random graph with no spatial constraint, the original reference.

Each condition was trained with five model seeds, and the two random conditions were run on eight different random topologies so we wouldn't depend on one lucky or unlucky wiring.

## The final result

```
hex_full             94.58%
hex_perm_full        84.21%   (gap to hex_full: 10.36 points)
local_random_full    81.44%   (gap to hex_full: 13.14 points)
global_random_full   83.21%   (gap to hex_full: 11.37 points)
```

The observation that settled it: the three non-hex conditions (hex_perm, local_random, global_random) sit close together at 81–84%, even though they differ radically in clustering and locality. hex_full stands alone, well above them at 94.58%.

So neither spatial locality nor high clustering explains the benefit. The only real source is consistent binding of weights to a fixed directional position: each direction (north, south, east, and so on) has its own weight matrix, shared across every node in the network.

## Is this a new discovery? No, and that is the honest part

This is exactly the principle that makes convolutional networks work, and it has been known for decades: each position in a convolution kernel has its own weight, and that is what lets CNNs beat plain networks on images. What we showed is that the same principle carries over to an irregular graph setting, where the structure is flexible rather than a rigid pixel grid. That's a genuinely different technical question from the original CNN one, but the result is a confirmation of a known principle, not an independent discovery.

## Limitations, stated plainly

- **Limited statistical power.** With eight random topologies per condition, the smallest possible permutation p-value is 0.1111 (the floor set by the sample size, not a sign of a weak effect). Hex beat 100% of all 16 random topologies, which is a strong signal, but reaching formal significance (p < 0.05) would take at least 19 random topologies.
- **Small scale.** 196 nodes on downsampled MNIST is far from any production setting.
- **Missing baselines.** We never ran a plain CNN or an MLP with a comparable budget. On a regular pixel grid, a CNN is the natural competitor and would very likely do at least as well, so this study says nothing about whether graph-based models are the right tool for this data.
- **Architecture sensitivity.** An independent reimplementation with a somewhat different gating design and a larger hidden size did not reproduce the same gap size. The benefit may shrink as model capacity grows, and we did not test that systematically.

## Closing

We set out to find a distinctive geometric shape, and we ended up confirming an old principle in a new setting. That is less exciting than we hoped at the start, but it is far more honest, and it was measured with a much sturdier method than anything earlier in this project. The lasting value isn't the final number. It's the habits we built along the way: a properly designed kill test, a real convergence protocol, and the reflex of asking "is this real, or is it an artifact of how I ran it?" before celebrating any result.
