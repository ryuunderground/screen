# Slide 17 figure — drawing spec (for GPT / image generation)

Slide 17 = **"Inference and Evaluation"**. The figure's job is to make **what happens to a NEW
ligand** immediately clear — a single left-to-right flow from an unseen molecule to a ranked
hit. (Evaluation metrics live in the slide's bullets, not in this figure.) Flat academic vector
style, navy (#1F3864) + gold (#C8A94B), white background, Arial-like font, wide.

## The flow — a NEW ligand, left → right (this is the whole figure)

1. **New ligand** — one unseen molecule pulled from a library (show a faint stack of other
   molecules behind it to signal "screening a library"). Label: "new molecule (no known pose)".

2. **→ Multi-channel scalar fields** — the ligand rendered as a small stack of field cubes
   (contact / H-bond / hydrophobic / …). Label: "render as fields".

3. **→ Pose search with w_pose** — icon: a score curve E(θ) with a **star at its peak**;
   the ligand snaps into the pocket. Label: "search poses (FFT + torsion) → predicted pose".
   This box uses the **pose weight set (w_pose)** — color it one accent (e.g. navy).

4. **→ Affinity score with w_affinity** — at that pose, read out a predicted binding strength
   (a small bar or number). Label: "score the pose → predicted affinity". This box uses the
   **affinity weight set (w_affinity)** — color it the other accent (e.g. gold).

5. **→ Ranked hits** — a sorted bar chart / list of library molecules, top few highlighted.
   Label: "rank the library → hit candidates".

### Must be obvious from the picture
- **Two weight sets, used in sequence:** w_pose (step 3, *where it binds*) then w_affinity
  (step 4, *how strongly*). Two distinct colors, two labeled boxes.
- The input is a **new, unseen** ligand with **no crystal pose** — the system produces the pose.

## Honesty note (small caption, keep it)
- Current validation is cognate redocking / a local funnel; **blind pose generation from
  scratch is future work.** Draw the flow, but don't imply blind docking is already solved.

## Style
- One clean horizontal pipeline, thin left→right arrows, rounded-rectangle blocks,
  2 accent colors only (navy = pose, gold = affinity), no photorealism, short labels.
