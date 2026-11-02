"""학회 발표용 덱 생성 — Introduction + Materials & Methods.

4:3, 남색 메인, 영어, 공식 발표용. 결과 수치는 넣지 않는다 (아직 없음; M&M까지만).
타 방법 비교는 정성적으로만 (미확인 수치 인용 금지 — 연구실 인식론 위생).

출력: slides/field_docking_intro_methods.pptx
"""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "slides" / "assets"


def img_aspect(path):
    im = Image.open(path); return im.size[0] / im.size[1]

# --- 팔레트 ---------------------------------------------------------------
NAVY = RGBColor(0x1F, 0x38, 0x64)      # 메인 남색
NAVY_DK = RGBColor(0x14, 0x24, 0x45)   # 짙은 남색 (배경)
GOLD = RGBColor(0xC8, 0xA9, 0x4B)      # 악센트 (절제해서)
INK = RGBColor(0x2B, 0x2B, 0x2B)       # 본문
GRAY = RGBColor(0x5A, 0x5A, 0x5A)      # 보조
LIGHT = RGBColor(0xF2, 0xF4, 0xF8)     # 연한 패널
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "Arial"
EMU_IN = 914400

prs = Presentation()
prs.slide_width = Inches(10)      # 4:3
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H = prs.slide_width, prs.slide_height

_page = 0


def _box(slide, x, y, w, h):
    return slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame


def _rect(slide, x, y, w, h, color):
    from pptx.enum.shapes import MSO_SHAPE
    sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.fill.solid(); sp.fill.fore_color.rgb = color
    sp.line.fill.background()
    sp.shadow.inherit = False
    return sp


def _set(p, text, size, color, bold=False, italic=False, align=PP_ALIGN.LEFT, font=FONT):
    p.alignment = align
    r = p.add_run(); r.text = text
    f = r.font
    f.size = Pt(size); f.bold = bold; f.italic = italic
    f.color.rgb = color; f.name = font
    return r


def footer(slide, tag):
    global _page
    _page += 1
    _rect(slide, 0.7, 7.02, 8.6, 0.015, NAVY)
    tf = _box(slide, 0.7, 7.05, 7.5, 0.3); tf.word_wrap = True
    _set(tf.paragraphs[0], "Field-Based Docking for Ultra-Large Virtual Screening",
         9, GRAY)
    tf2 = _box(slide, 8.4, 7.05, 0.9, 0.3)
    p = tf2.paragraphs[0]; p.alignment = PP_ALIGN.RIGHT
    _set(p, f"{_page:02d}  ·  {tag}", 9, GRAY)


# --- 슬라이드 종류 --------------------------------------------------------
def title_slide(title, subtitle, authors):
    s = prs.slides.add_slide(BLANK)
    _rect(s, 0, 0, 10, 7.5, NAVY_DK)
    _rect(s, 0.9, 2.55, 1.4, 0.07, GOLD)
    tf = _box(s, 0.9, 2.75, 8.2, 2.0); tf.word_wrap = True
    _set(tf.paragraphs[0], title, 36, WHITE, bold=True)
    p = tf.add_paragraph(); p.space_before = Pt(14)
    _set(p, subtitle, 19, RGBColor(0xC9, 0xD2, 0xE6), italic=True)
    tf2 = _box(s, 0.9, 5.5, 8.2, 1.0); tf2.word_wrap = True
    _set(tf2.paragraphs[0], authors, 14, RGBColor(0xB8, 0xC2, 0xDA))
    p = tf2.add_paragraph(); p.space_before = Pt(4)
    _set(p, "Research design — Introduction & Methods", 12, GOLD)
    return s


def section_slide(number, name):
    s = prs.slides.add_slide(BLANK)
    _rect(s, 0, 0, 10, 7.5, NAVY)
    tf = _box(s, 0.9, 3.0, 8.2, 1.2)
    _set(tf.paragraphs[0], number, 20, GOLD, bold=True)
    tf2 = _box(s, 0.9, 3.5, 8.2, 1.2)
    _set(tf2.paragraphs[0], name, 40, WHITE, bold=True)
    _rect(s, 0.9, 3.42, 1.1, 0.06, GOLD)
    return s


def _pic(slide, name, x, y, w):
    """이미지를 폭 w(in)로, 비율 유지해 배치. 반환 높이(in)."""
    path = str(ASSETS / name) if (ASSETS / name).exists() else str(ROOT / name)
    asp = img_aspect(path)
    h = w / asp
    slide.shapes.add_picture(path, Inches(x), Inches(y), width=Inches(w))
    return h


def content_slide(title, bullets, tag, kicker=None, note=None,
                  image=None, img_mode="right", img_w=8.6, bsize=17, bsub=14.5):
    """bullets: [(text, level)]. image: 파일명. img_mode: 'right' | 'bottom'."""
    s = prs.slides.add_slide(BLANK)
    y = 0.6
    if kicker:
        tf = _box(s, 0.72, y, 8.6, 0.35)
        _set(tf.paragraphs[0], kicker.upper(), 11, GOLD, bold=True)
        y += 0.42
    tf = _box(s, 0.7, y, 8.6, 1.0); tf.word_wrap = True
    _set(tf.paragraphs[0], title, 27, NAVY, bold=True)
    _rect(s, 0.72, y + 0.72, 1.0, 0.05, GOLD)

    bw = 4.5 if (image and img_mode == "right") else 8.55
    body = _box(s, 0.72, y + 1.0, bw, 4.6)
    body.word_wrap = True
    first = True
    for text, lvl in bullets:
        p = body.paragraphs[0] if first else body.add_paragraph()
        first = False
        if lvl == 0:
            p.space_before = Pt(11)
            _set(p, "▪  ", 15, GOLD, bold=True)
            _set(p, text, bsize, INK)
        else:
            p.space_before = Pt(5)
            p.level = 1
            _set(p, "–  ", 13, GRAY)
            _set(p, text, bsub, GRAY)

    if image:
        if img_mode == "right":
            _pic(s, image, 5.35, y + 1.15, 3.95)
        else:  # bottom, 가운데
            asp = img_aspect(str(ASSETS / image) if (ASSETS / image).exists()
                             else str(ROOT / image))
            iw = img_w; ih = iw / asp
            top = 6.82 - ih
            _pic(s, image, (10 - iw) / 2, top, iw)
    if note:
        nb = _box(s, 0.72, 6.5, 8.55, 0.5); nb.word_wrap = True
        _set(nb.paragraphs[0], note, 12, NAVY, italic=True)
    footer(s, tag)
    return s


def pipeline_slide(title, tag, kicker):
    """2단계 파이프라인 도식."""
    s = prs.slides.add_slide(BLANK)
    tf = _box(s, 0.72, 0.6, 8.6, 0.4)
    _set(tf.paragraphs[0], kicker.upper(), 11, GOLD, bold=True)
    tf = _box(s, 0.7, 1.0, 8.6, 0.9); tf.word_wrap = True
    _set(tf.paragraphs[0], title, 27, NAVY, bold=True)
    _rect(s, 0.72, 1.72, 1.0, 0.05, GOLD)

    def stage(x, w, head, lines, fill, headcol):
        _rect(s, x, 2.3, w, 2.5, fill)
        tf = _box(s, x + 0.2, 2.45, w - 0.4, 0.6)
        _set(tf.paragraphs[0], head, 15, headcol, bold=True)
        bt = _box(s, x + 0.2, 3.05, w - 0.4, 1.7); bt.word_wrap = True
        f = True
        for ln in lines:
            p = bt.paragraphs[0] if f else bt.add_paragraph(); f = False
            p.space_before = Pt(6)
            _set(p, ln, 12.5, INK if fill == LIGHT else WHITE)

    stage(0.72, 3.9, "① Cheap screen", [
        "10⁹ → 10⁶ compounds",
        "Rotation-invariant field summary",
        "No pose; fast nearest-neighbor"], LIGHT, NAVY)
    # 화살표
    at = _box(s, 4.75, 3.3, 0.6, 0.6)
    _set(at.paragraphs[0], "→", 32, GOLD, bold=True, align=PP_ALIGN.CENTER)
    stage(5.38, 3.9, "② Precise dock", [
        "10⁶ survivors",
        "Full FFT correlation landscape",
        "Pose + score; internal torsion"], NAVY, GOLD)

    band = _box(s, 0.72, 5.05, 8.55, 0.9); band.word_wrap = True
    _set(band.paragraphs[0], "Both stages share one multi-channel field representation — ",
         14, INK, bold=True)
    _set(band.paragraphs[0], "no separate learned tower.", 14, NAVY, bold=True, italic=True)
    footer(s, tag)
    return s


def table_slide(title, headers, rows, tag, kicker=None, note=None):
    s = prs.slides.add_slide(BLANK)
    y = 0.6
    if kicker:
        tf = _box(s, 0.72, y, 8.6, 0.35)
        _set(tf.paragraphs[0], kicker.upper(), 11, GOLD, bold=True); y += 0.42
    tf = _box(s, 0.7, y, 8.6, 1.0); tf.word_wrap = True
    _set(tf.paragraphs[0], title, 27, NAVY, bold=True)
    _rect(s, 0.72, y + 0.72, 1.0, 0.05, GOLD)

    nr, nc = len(rows) + 1, len(headers)
    th = 0.62 + 0.72 * len(rows)
    tbl = s.shapes.add_table(nr, nc, Inches(0.72), Inches(y + 1.15),
                             Inches(8.56), Inches(th)).table
    tbl.first_row = False; tbl.horz_banding = False
    tbl.columns[0].width = Inches(2.3)
    for c in range(1, nc):
        tbl.columns[c].width = Inches((8.56 - 2.3) / (nc - 1))
    for c, htxt in enumerate(headers):
        cell = tbl.cell(0, c); cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
        p = cell.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        _set(p, htxt, 13, WHITE, bold=True)
    for r, row in enumerate(rows, 1):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c); cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT if r % 2 else WHITE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            _set(p, val, 12.5, NAVY if c == 0 else INK, bold=(c == 0))
    if note:
        nb = _box(s, 0.72, 6.55, 8.55, 0.5); nb.word_wrap = True
        _set(nb.paragraphs[0], note, 12, NAVY, italic=True)
    footer(s, tag)
    return s


# =========================================================================
title_slide(
    "From Points to Curves",
    "Field-based docking with an FFT scoring landscape and internal torsion",
    "Author · Affiliation")

# ---- Part I: Introduction ------------------------------------------------
section_slide("Part I", "Introduction")

content_slide(
    "Molecular docking in drug discovery",
    [("Docking predicts how a small molecule binds a protein target", 0),
     ("Two outputs: the binding pose (3D geometry) and a score", 1),
     ("Backbone of structure-based virtual screening", 0),
     ("A good score separates true binders from the non-binding majority", 1)],
    "Background", kicker="Why docking")

content_slide(
    "The scaling wall: billion-compound libraries",
    [("Make-on-demand libraries now exceed 10⁹ molecules", 0),
     ("Classical docking is seconds to minutes per compound", 1),
     ("Exhaustively docking a billion molecules is infeasible", 0),
     ("Screening must be staged: cheap filters first, careful docking last", 0)],
    "Problem", kicker="The scaling wall",
    image="funnel.png", img_mode="right")

content_slide(
    "Structure-based virtual screening today",
    [("Expert sets up each target: pocket, protonation, parameters", 0),
     ("Candidate poses are often inspected by hand", 1),
     ("Throughput is the bottleneck, not ideas", 0),
     ("We want a fast pre-screen that still respects 3D geometry", 0)],
    "Context", kicker="How it is done now")

content_slide(
    "Fast screening — approach 1: contrastive embeddings",
    [("Embed the pocket and each ligand as a vector", 0),
     ("Binding ≈ similarity; precompute once, scan 10⁹ by dot product", 1),
     ("Very fast — this is its strength", 0),
     ("Limitation: the interaction collapses to a single point", 0),
     ("No pose comes out; geometry is discarded", 1)],
    "Prior art", kicker="Approach 1 — the point")

content_slide(
    "Fast screening — approach 2: generative docking",
    [("Diffusion models denoise from noise to a 3D pose", 0),
     ("Strength: produces poses, competitive on redocking", 1),
     ("Internal torsional degrees of freedom are hard", 0),
     ("Flexible bonds are avoided or not modeled explicitly", 1),
     ("Still a per-molecule generator, not a cheap invariant screen", 1)],
    "Prior art", kicker="Approach 2 — denoising")

table_slide(
    "Where each approach falls short",
    ["", "Contrastive", "Generative docking", "Ours (curve)"],
    [["Gives a pose?", "No", "Yes", "Yes"],
     ["Internal torsion", "n/a", "Limited / avoided", "Yes (fragments)"],
     ["Billion-scale screen", "Yes (fast)", "Costly per molecule", "Yes (staged)"],
     ["Score basis", "Learned only", "Learned only", "Physical channels"]],
    "Positioning", kicker="Synthesis",
    note="We keep the pose and the torsion a point loses, while staying cheap enough to screen.")

content_slide(
    "Our key idea: from a point to a curve",
    [("Represent each molecule as a field, not a point", 0),
     ("Binding = field overlap over pose → a curve, not a scalar", 0),
     ("Its peak gives pose AND score; FFT draws the whole curve cheaply", 0)],
    "Contribution", kicker="This work",
    image="point_vs_curve.png", img_mode="bottom", img_w=8.2, bsize=15.5)

content_slide(
    "Why a field? overlap is rotation-equivariant",
    [("A field moves with the molecule it describes", 0),
     ("Rotate the system → the field rotates → overlap score is preserved", 1),
     ("The right object for scoring geometry, not a pose-blind summary", 0)],
    "Rationale", kicker="Why fields",
    image="equivariance.png", img_mode="bottom", img_w=7.4, bsize=15.5)

content_slide(
    "Contributions of this work",
    [("A field-overlap scoring landscape — pose and score from one curve", 0),
     ("FFT over translations; orientations as the outer search", 1),
     ("Internal torsion via fragment decomposition with a data-driven cut", 0),
     ("A two-stage screen that scales toward 10⁹, sharing one representation", 0),
     ("Evaluation that includes held-out enrichment, not only redocking", 0)],
    "Claims", kicker="What we contribute")

# ---- Part II: Materials & Methods ----------------------------------------
section_slide("Part II", "Materials & Methods")

pipeline_slide("Two-stage screening pipeline", "Overview", kicker="System design")

content_slide(
    "Field representation and channels",
    [("Receptor and ligand rendered as multi-channel fields on a grid", 0),
     ("Channels: shape, polar, charge, hydrophobic", 1),
     ("Score = weighted sum of channel overlaps (linear → FFT-able)", 0)],
    "Representation", kicker="Methods",
    image="field_channels.png", img_mode="bottom", img_w=7.0, bsize=15.5)

content_slide(
    "From atoms to fields: grid rendering",
    [("Each atom contributes a smooth Gaussian to its channels", 0),
     ("Fields live on a regular grid at a chosen resolution", 1),
     ("Smooth and differentiable — no brittle atom-pair cutoffs", 0),
     ("Resolution trades accuracy against grid size and FFT cost", 1)],
    "Representation", kicker="Methods")

content_slide(
    "Scoring: the correlation landscape via FFT",
    [("For a fixed orientation, one FFT gives overlap at every translation", 0),
     ("The entire translational landscape in a single transform", 1),
     ("Global maximum over orientation and translation → pose + score", 0),
     ("E(t) = Σ_channels  w · (receptor ⋆ ligand)(t)", 1),
     ("FFT is what makes the full curve affordable", 0)],
    "Scoring", kicker="Methods",
    image="corr_landscape.png", img_mode="right")

content_slide(
    "Searching orientations over SO(3)",
    [("FFT handles all translations at once; orientation is the outer loop", 0),
     ("Sample rotations over SO(3); one FFT landscape per rotation", 1),
     ("Keep the global maximum across the sampled orientations", 0),
     ("Rotations and torsion are the real cost — translations are cheap", 1)],
    "Search", kicker="Methods")

content_slide(
    "Internal torsion by fragment decomposition",
    [("Cut only where a bond moves atoms substantially", 0),
     ("Data-driven threshold from ligand geometry alone, offline", 1),
     ("Each rigid fragment scored by FFT", 0),
     ("Fragments stitched on a kinematic tree — one dihedral per step", 0),
     ("Score stays accurate with chain depth (validated)", 1)],
    "Flexibility", kicker="Methods",
    image="fragment_tree.png", img_mode="right")

content_slide(
    "Preliminary analysis: the torsion response is smooth",
    [("Swept each rotatable bond; measured the interaction-score curve", 0),
     ("Curves are smooth — a few Fourier modes capture them", 1),
     ("Terminal bonds move little; only core bonds need cutting", 0),
     ("Motivates the selective cut criterion above", 1)],
    "Evidence", kicker="Preliminary (prior analysis)",
    image="out/torsion_spectrum.png", img_mode="bottom", img_w=9.0, bsize=15.5)

content_slide(
    "Learning: channel weights, not the docker",
    [("The docking engine stays physics-based — poses must be trustworthy", 0),
     ("Only the channel weights (and the cheap screening summary) are learned", 1),
     ("Small, interpretable model on top of a mechanistic core", 1),
     ("Labels: experimental binding affinity (PDBbind Kᵈ/Kᵢ)", 0),
     ("Not docking scores — training on those would cap us at their accuracy", 1)],
    "Learning", kicker="Methods")

content_slide(
    "Datasets and cluster-aware cross-validation",
    [("Training and model selection: PDBbind — poses and affinities", 0),
     ("5-fold CV, grouped by protein and ligand-scaffold cluster", 1),
     ("Cluster grouping prevents leakage from near-duplicate complexes", 0)],
    "Data", kicker="Methods",
    image="cv_split.png", img_mode="bottom", img_w=7.6, bsize=15.5)

content_slide(
    "Evaluation: benchmarks and metrics",
    [("Pose accuracy — PoseBusters v2, redocking RMSD < 2 Å", 0),
     ("Comparable to the split recent generative dockers report", 1),
     ("Enrichment — LIT-PCBA, EF and BEDROC on held-out targets", 0),
     ("Held-out enrichment is what prior field-based methods omit", 1)],
    "Evaluation", kicker="Methods",
    note="Research design — Introduction and Methods. Results to follow once trained.")

out = Path(__file__).resolve().parent.parent / "slides" / "field_docking_intro_methods.pptx"
out.parent.mkdir(exist_ok=True)
prs.save(str(out))
print(f"saved: {out}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
