# Overview schematic — drawing spec (for GPT / image generation)

Slide 11 다음(M&M 시작부)에 들어갈 **전체 파이프라인 모식도** 1장. 아래를 그대로 이미지 생성 프롬프트로 쓰면 됨.

## One-line intent
A single horizontal pipeline that shows: protein + ligand → multi-channel scalar fields →
field-overlap score E → used two ways (affinity weights / pose weights) → outputs (binding
strength / predicted pose). Clean, academic, flat vector style. Navy (#1F3864) + gold (#C8A94B),
white background, Arial-like font. 4:3, wide.

## Blocks (left → right)

1. **Input**
   - Protein pocket (cartoon) + ligand (sticks), labeled "Receptor" / "Ligand".

2. **Representation — multi-channel scalar fields**
   - Each molecule → a stack of 3D field cubes, one per interaction channel:
     contact/shape, clash, hydrophobic, H-bond, electrostatics, desolvation.
   - Label: "Each molecule → multi-channel scalar fields (R³→scalar)".

3. **Score E (field overlap)**
   - Overlap the two field stacks → one number per pose.
   - Formula box: **E = Σᵢ wᵢ · overlapᵢ**  (weighted sum of per-channel field overlaps).

4. **Two weight sets (same channels, different wᵢ)** — branch into two arrows:
   - **w_affinity** → "Binding strength" (trained on PDBbind experimental affinity).
   - **w_pose** → "Pose search" (trained so the native pose scores highest, by ranking
     native > decoys on RMSD).

5. **Pose search (FFT + torsion)**
   - Under w_pose: rotate/translate + internal torsion → curve E(θ); peak = predicted pose.
   - Small icon: a curve with a star at its peak; "FFT sweeps translation/rotation, torsion = extra axis".

6. **Outputs**
   - Predicted pose (ligand overlaid in pocket) + predicted binding strength (a number / bar).
   - For screening: rank a library by predicted affinity.

## Supervision (small footnote band under the pipeline)
- **Affinity task:** E_affinity ≈ experimental pKd (PDBbind).
- **Pose task:** E_pose ranks native > decoys (supervised on RMSD, not affinity).
- Note: only the weights wᵢ are learned; the channel functions are fixed physics (Vina/Glide-style).

## Honesty note (optional small text)
- Current scope: cognate redocking / local funnel. **Blind global docking = future work.**

## Style
- Flat vector, thin arrows left→right, rounded rectangle blocks, 2 accent colors only.
- No photorealism. Labels short. The two-weight branch should be visually obvious (two arrows).
