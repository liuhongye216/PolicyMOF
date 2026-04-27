"""
Attention Visualization for Paper Publication

This script generates publication-quality attention heatmaps and related figures.
Designed for academic papers with proper sizing, fonts, and color schemes.

Key improvements over the original:
1. Publication-ready figure sizes (single/double column)
2. YlOrRd colormap for attention (warm colors = high attention)
3. DPI=300 for print quality
4. Selective layer/head visualization (not all 32)
5. Proper font sizes for readability after print
6. Colorbar with labels
"""

import argparse
import os
import json
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from typing import Optional, List, Tuple, Dict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr, spearmanr

# ============================================================================
# Nature Publication style settings
# Reference: Nature Author Guidelines - Figure & Graphics Requirements
# - Font: Helvetica / Arial (sans-serif)
# - Min font size after scaling: 5 pt; recommended 7–8 pt
# - Single column width: 89 mm (3.50 in)
# - Double column width: 183 mm (7.20 in)
# - 1.5 column width: 120–136 mm (~5.31 in)
# - Max depth: 247 mm (9.72 in)
# - Resolution: ≥300 DPI (colour/halftone), ≥600 DPI (line art)
# - Min line width: 0.5 pt
# - Panel labels: bold lowercase a, b, c, d (NO parentheses)
# - Format: PDF/EPS (vector preferred), TIFF, PNG acceptable
# ============================================================================
# ----------------------------------------------------------------------------
# Typography policy (user-requested):
#   * Font family: Arial (sans-serif), NEVER bold.
#   * Body size  : BODY_FS = 7 pt  (Nature body-text equivalent).
#   * Reduced    : SMALL_FS = 6 pt used ONLY when a label is too dense to fit
#                  at 7 pt (e.g. many-token tick labels, dense legends).
#   * Tiny       : TINY_FS  = 5 pt reserved for EXTREME density (small panels
#                  with >20 rotated tokens) — used sparingly as last resort.
# ----------------------------------------------------------------------------
BODY_FS = 7
SMALL_FS = 6
TINY_FS = 5
# Panel letter labels (a, b, c, d) are 1 pt above body size so the reader
# can tell them apart from in-axes text at a glance (matches Nature figures).
PANEL_FS = 8

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': BODY_FS,
    'font.weight': 'normal',
    'axes.titlesize': BODY_FS,
    'axes.titleweight': 'normal',
    'axes.labelsize': BODY_FS,
    'axes.labelweight': 'normal',
    'xtick.labelsize': BODY_FS,
    'ytick.labelsize': BODY_FS,
    'legend.fontsize': BODY_FS,
    'figure.titlesize': BODY_FS,
    'figure.titleweight': 'normal',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    'lines.linewidth': 0.8,
    'patch.linewidth': 0.5,
    'mathtext.default': 'regular',
    'pdf.fonttype': 42,    # embed TrueType (editable text in PDF)
    'ps.fonttype': 42,
})

# Figure widths for Nature publication (inches)
SINGLE_COL_WIDTH = 3.50    # Nature single column: 89 mm (3.50 in)
DOUBLE_COL_WIDTH = 7.20    # Nature double column: 183 mm (7.20 in)
ONE_HALF_COL_WIDTH = 5.31  # Nature 1.5 column: ~136 mm (5.31 in)
MAX_HEIGHT = 9.72           # Nature max figure depth: 247 mm (9.72 in)
DPI = 300                   # Nature minimum for colour figures
SAVE_FORMATS = ['png', 'tiff', 'pdf']  # PNG (review), TIFF & PDF (submission)


def load_model_and_tokenizer(
    model_path: str,
    adapter_path: str = None,
    device: str = "cpu"
):
    """Load model and tokenizer with attention output enabled."""
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    num_labels = 6
    if adapter_path:
        args_path = os.path.join(adapter_path, "args.json")
        if os.path.exists(args_path):
            with open(args_path, 'r') as f:
                train_args = json.load(f)
            num_labels = train_args.get('num_labels', 6)

    print(f"Loading base model from: {model_path}")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        output_attentions=True,
        output_hidden_states=True,
    )

    if adapter_path:
        print(f"Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model

    model = model.to(device)
    model.eval()
    return model, tokenizer


def get_attention_weights(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    text: str,
    device: str = "cpu",
    exclude_special_tokens: bool = True
) -> Tuple[torch.Tensor, List[str], torch.Tensor, List[int]]:
    """Get attention weights for the input text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)

    attentions = outputs.attentions
    hidden_states = outputs.hidden_states

    all_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    special_tokens = {
        '<s>', '</s>', '<pad>', '[CLS]', '[SEP]', '[PAD]',
        '<|endoftext|>', '<|begin_of_text|>', '<|end_of_text|>',
        '[BOS]', '[EOS]', '<bos>', '<eos>', '<unk>',
        'Ġ', '▁',  # BPE/SentencePiece space tokens (no chemical meaning)
    }

    if exclude_special_tokens:
        valid_indices = [i for i, t in enumerate(all_tokens) if t not in special_tokens]
        tokens = [all_tokens[i] for i in valid_indices]

        filtered_attentions = []
        for attn in attentions:
            valid_idx_tensor = torch.tensor(valid_indices, device=attn.device)
            filtered = attn[:, :, valid_idx_tensor, :][:, :, :, valid_idx_tensor]
            filtered_attentions.append(filtered)
        attentions = tuple(filtered_attentions)
    else:
        tokens = all_tokens
        valid_indices = list(range(len(all_tokens)))

    return attentions, tokens, hidden_states, valid_indices


def _clean_token_label(token: str) -> str:
    """Clean token for display (remove sentencepiece/BPE artifacts)."""
    return token.replace('▁', '').replace('Ġ', ' ').strip()


# ============================================================================
# SMILES Token Classification
# ============================================================================
# Known MOFid topology tokens (extend as needed for your dataset)
_MOFID_TOPOLOGY_TOKENS = {
    'pcu', 'sql', 'fcu', 'bcu', 'dia', 'sra', 'nbo', 'pts', 'acs', 'tbo',
    'she', 'rht', 'ftw', 'the', 'ith', 'sod', 'nia', 'lvt', 'qtz', 'cds',
    'hxg', 'bnn', 'kgm', 'hcb', 'pto', 'pyr', 'spn', 'ssa', 'ssb',
}
# MOFid category tokens (cat0–cat9)
_MOFID_CAT_PATTERN = {'cat0', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5',
                       'cat6', 'cat7', 'cat8', 'cat9'}

_SPECIAL_TOKENS = {
    '<s>', '</s>', '<pad>', '[CLS]', '[SEP]', '[PAD]',
    '<|endoftext|>', '<|begin_of_text|>', '<|end_of_text|>',
    '[BOS]', '[EOS]', '<bos>', '<eos>', '<unk>', 'Ġ', '▁',
}

# SMILES aromatic atoms: lowercase letters only (case-sensitive!)
_AROMATIC_ATOMS = {'c', 'n', 'o', 's', 'p', 'se', 'as'}

# Ring closure digits (single-char only; SMILES uses '1'–'9')
_RING_DIGITS = {'1', '2', '3', '4', '5', '6', '7', '8', '9'}

# Branch tokens
_BRANCH_TOKENS = {'(', ')'}

# Bond tokens (explicit bonds in SMILES)
_BOND_TOKENS = {'=', '#', '/', '\\', '-'}

# Metal atoms typically in square brackets
_METAL_PATTERN = re.compile(r'^\[([A-Z][a-z]?)\]$|^\[([A-Z][a-z]?)[\d\+\-]')
_COMMON_METALS = {
    'Cu', 'Zn', 'Fe', 'Co', 'Ni', 'Mn', 'Cr', 'V', 'Ti', 'Cd', 'Pb',
    'Ag', 'Au', 'Pt', 'Pd', 'Ir', 'Ru', 'Rh', 'Os', 'Mo', 'W', 'Zr',
    'Hf', 'Al', 'Ga', 'In', 'Sn', 'Bi', 'Sc', 'Y', 'La', 'Ce', 'Nd',
    'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Li', 'Na', 'K', 'Rb', 'Cs', 'Be', 'Mg', 'Ca', 'Sr', 'Ba',
}


def _classify_smiles_token(token: str) -> str:
    """
    Classify a single SMILES/MOFid token into a chemical feature category.

    Categories:
        'MOFid'        – MOFid version tags, topology codes, category markers, separator dots
        'Metal'        – Square-bracket metal atoms, e.g. [Cu], [Zn2+]
        'Ring Closures' – Pure ring-closure digit tokens ('1', '2', …, '9')
        'Branches'     – '(' or ')'
        'Bonds'        – Explicit bond symbols ('=', '#', '-', '/', '\\')
        'Aromatic'     – Lowercase aromatic atoms in SMILES ('c', 'n', 'o', 's', …)
        'Atoms'        – Non-aromatic atoms (uppercase C, N, O, S, charged groups like [O-])
        'Other'        – Anything else

    Design principles:
      1. MOFid tokens are checked first (they may contain digits, hyphens, etc.)
      2. Case-sensitive: 'c' = aromatic carbon, 'C' = aliphatic carbon
      3. Multi-character tokens like 'CC', 'N#C' are classified by their
         dominant chemical role, not by substring matching of individual chars
    """
    if token in _SPECIAL_TOKENS:
        return 'Special'

    clean = token.replace('▁', '').replace('Ġ', '')
    if not clean:
        return 'Special'

    # --- 1. MOFid tokens (must be checked before anything else) ---
    if clean.startswith('MOFid'):
        return 'MOFid'
    if clean.lower() in _MOFID_TOPOLOGY_TOKENS:
        return 'MOFid'
    if clean.lower() in _MOFID_CAT_PATTERN:
        return 'MOFid'

    # --- 2. Metal atoms in square brackets: [Cu], [Zn], [Fe2+] etc. ---
    m = _METAL_PATTERN.match(clean)
    if m:
        atom_symbol = m.group(1) or m.group(2)
        if atom_symbol in _COMMON_METALS:
            return 'Metal'

    # --- 3. Pure ring-closure digits ---
    # Only classify if the entire token is a single digit
    if clean in _RING_DIGITS:
        return 'Ring Closures'

    # --- 4. Branch tokens ---
    if clean in _BRANCH_TOKENS:
        return 'Branches'

    # --- 5. Pure bond tokens ---
    if clean in _BOND_TOKENS:
        return 'Bonds'

    # --- 6. Aromatic atoms (case-sensitive: lowercase only) ---
    # Single-char aromatic: 'c', 'n', 'o', 's'
    if clean in _AROMATIC_ATOMS:
        return 'Aromatic'

    # --- 7. Non-aromatic atoms / functional groups ---
    # Uppercase single atom: C, N, O, S, P, etc.
    # Charged groups: [O-], [NH4+], etc.
    # Multi-atom tokens: CC, N#C, C#N, etc. — these are atom/bond combos
    # Separator dots in SMILES (fragment separator)
    if clean == '.':
        return 'Other'

    return 'Atoms'


def _categorize_tokens(
    tokens: List[str],
    importance: np.ndarray,
    categories_order: List[str] = None,
) -> Dict[str, list]:
    """
    Classify a list of tokens and group their attention values by category.

    Returns dict  {category_name: [attention_values]}.
    """
    if categories_order is None:
        categories_order = [
            'Ring Closures', 'Branches', 'Bonds',
            'Aromatic', 'Metal', 'MOFid', 'Atoms', 'Other',
        ]
    cats = {c: [] for c in categories_order}

    for idx, token in enumerate(tokens):
        label = _classify_smiles_token(token)
        if label == 'Special':
            continue
        # Map to one of the requested categories; fall through to 'Atoms'
        if label in cats:
            cats[label].append(importance[idx])
        elif 'Atoms' in cats:
            cats['Atoms'].append(importance[idx])
        # else silently skip

    # Remove empty categories so the bar chart only shows populated ones
    return {k: v for k, v in cats.items() if v}


def _save_figure(fig, output_path: str):
    """Save figure in multiple formats (PNG for review, TIFF & PDF for Nature submission)."""
    for fmt in SAVE_FORMATS:
        path = output_path.rsplit('.', 1)[0] + f'.{fmt}'
        save_dpi = 600 if fmt == 'tiff' else DPI
        fig.savefig(path, dpi=save_dpi, bbox_inches='tight', pad_inches=0.02,
                    facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {output_path} (formats: {SAVE_FORMATS})")


# ============================================================================
# Figure 1: Average Attention Heatmap (self-attention, all layers averaged)
# ============================================================================
def plot_avg_attention_heatmap(
    attentions: Tuple[torch.Tensor],
    tokens: List[str],
    output_path: str,
    title: str = None,
    max_tokens: int = 40,
):
    """
    ACS publication-quality average attention heatmap.
    If tokens > max_tokens, truncate for readability.
    """
    all_attn = torch.stack([attn[0] for attn in attentions])
    avg_attn = all_attn.mean(dim=(0, 1)).float().cpu().numpy()

    display_tokens = [_clean_token_label(t) for t in tokens]

    # Truncate if too many tokens
    if len(display_tokens) > max_tokens:
        avg_attn = avg_attn[:max_tokens, :max_tokens]
        display_tokens = display_tokens[:max_tokens]

    n = len(display_tokens)

    # ACS double column width; height capped at MAX_HEIGHT
    fig_height = min(DOUBLE_COL_WIDTH * 0.85, MAX_HEIGHT)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, fig_height))

    im = ax.imshow(avg_attn, cmap='YlOrRd', aspect='equal', interpolation='nearest')

    # Tick labels: many tokens rotated 90°; use SMALL_FS so they don't overlap
    tick_fs = SMALL_FS if n > 25 else BODY_FS
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(display_tokens, rotation=90, ha='center', fontsize=tick_fs)
    ax.set_yticklabels(display_tokens, fontsize=tick_fs)

    ax.set_xlabel('Key Tokens', fontsize=BODY_FS, labelpad=6)
    ax.set_ylabel('Query Tokens', fontsize=BODY_FS, labelpad=6)

    if title:
        ax.set_title(title, fontsize=BODY_FS, pad=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.03, aspect=30)
    cbar.set_label('Attention Weight', fontsize=BODY_FS)
    cbar.ax.tick_params(labelsize=BODY_FS, width=0.5)
    cbar.outline.set_linewidth(0.5)

    plt.tight_layout()
    _save_figure(fig, output_path)


# ============================================================================
# Figure 2: Token Importance Bar Chart
# ============================================================================
# ...existing code...
def plot_token_importance(
    attentions: Tuple[torch.Tensor],
    tokens: List[str],
    output_path: str,
    title: str = None,
):
    """ACS publication-quality token importance bar chart.
    
    Token importance is defined as the column-wise mean of the averaged
    attention matrix (i.e., how much attention each token receives on average
    from all query positions).  This definition is consistent with the
    cross-layer heatmap (panel d) and the combined figure (panel a).
    """
    all_attn = torch.stack([attn[0] for attn in attentions])
    avg_attn_matrix = all_attn.mean(dim=(0, 1)).float().cpu().numpy()  # (seq, seq)
    token_importance = avg_attn_matrix.mean(axis=0)  # column-wise mean -> (seq,)

    display_tokens = [_clean_token_label(t) for t in tokens]
    n = len(display_tokens)

    # Normalize for color mapping
    token_importance_norm = token_importance / token_importance.max()

    # ACS double column, moderate height
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.3))

    colors = plt.cm.YlOrRd(token_importance_norm)
    x = np.arange(n)

    ax.bar(x, token_importance, color=colors, edgecolor='grey', linewidth=0.5, width=0.8)

    tick_fs = SMALL_FS if n > 25 else BODY_FS
    ax.set_xticks(x)
    ax.set_xticklabels(display_tokens, rotation=90, ha='center', fontsize=tick_fs)
    ax.set_ylabel('Avg Attn Received', fontsize=BODY_FS)
    ax.tick_params(axis='y', labelsize=BODY_FS)
    ax.set_xlim(-0.5, n - 0.5)

    if title:
        ax.set_title(title, fontsize=BODY_FS, pad=6)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    _save_figure(fig, output_path)

# ============================================================================
# Figure 3: First Token (CLS) Attention Across Layers
# ============================================================================
# ...existing code...
def plot_first_token_attention(
    attentions: Tuple[torch.Tensor],
    tokens: List[str],
    output_path: str,
    title: str = None,
    select_layers: List[int] = None,
):
    """
    Heatmap: per-token attention importance across selected layers.
    
    For decoder-only models (e.g., LLaMA) with causal masking, the first
    token can only attend to itself, producing a trivial all-zero pattern
    for positions > 0.  Instead, we compute the **column-wise mean** of the
    attention matrix at each layer (i.e., how much attention each token
    *receives* on average from all query positions).  This is equivalent to
    the per-layer token importance and produces a meaningful cross-layer
    heatmap for any architecture.
    
    Each layer is independently normalized to [0, 1] (row-wise normalization)
    so that the colormap reveals intra-layer contrast even when absolute
    magnitudes are similar across layers.
    """
    num_layers = len(attentions)

    if select_layers is None:
        if num_layers <= 8:
            select_layers = list(range(num_layers))
        else:
            step = num_layers // 6
            select_layers = sorted(set([0, step, 2*step, num_layers//2, 
                                         num_layers - step, num_layers - 2, num_layers - 1]))

    display_tokens = [_clean_token_label(t) for t in tokens]

    attention_matrix = []
    layer_labels = []
    for li in select_layers:
        # Column-wise mean: average attention *received* by each token
        # Shape: mean over heads -> (seq, seq), then mean over rows (queries)
        layer_attn = attentions[li][0].mean(dim=0).float().cpu().numpy()  # (seq, seq)
        token_received = layer_attn.mean(axis=0)  # (seq,)
        attention_matrix.append(token_received)
        layer_labels.append(f'Layer {li}')

    attention_matrix = np.array(attention_matrix)

    # === Row-wise normalization: each layer independently scaled to [0, 1] ===
    row_min = attention_matrix.min(axis=1, keepdims=True)
    row_max = attention_matrix.max(axis=1, keepdims=True)
    row_range = row_max - row_min
    row_range[row_range == 0] = 1.0  # avoid division by zero for uniform layers
    attention_matrix_norm = (attention_matrix - row_min) / row_range

    n_tokens = len(display_tokens)

    fig_height = min(max(2.0, len(select_layers) * 0.35 + 1.0), MAX_HEIGHT * 0.4)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, fig_height))

    im = ax.imshow(attention_matrix_norm, cmap='YlOrRd', aspect='auto', interpolation='nearest')

    # Many token labels rotated 90° are dense; shrink only when needed
    tick_fs = SMALL_FS if n_tokens > 25 else BODY_FS
    ax.set_xticks(np.arange(n_tokens))
    ax.set_xticklabels(display_tokens, rotation=90, ha='center', fontsize=tick_fs)
    ax.set_yticks(np.arange(len(layer_labels)))
    ax.set_yticklabels(layer_labels, fontsize=BODY_FS)

    ax.set_xlabel('Token', fontsize=BODY_FS, labelpad=6)
    ax.set_ylabel('Layer', fontsize=BODY_FS, labelpad=6)

    if title:
        ax.set_title(title, fontsize=BODY_FS, pad=6)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03, aspect=25)
    cbar.set_label('Normalized Attn (per-layer)', fontsize=BODY_FS)
    cbar.ax.tick_params(labelsize=BODY_FS, width=0.5)
    cbar.outline.set_linewidth(0.5)

    plt.tight_layout()
    _save_figure(fig, output_path)

# ============================================================================
# Figure 4: Selected Layer Attention Patterns (2×2 or 2×3 grid)
# ============================================================================
def plot_selected_layer_patterns(
    attentions: Tuple[torch.Tensor],
    tokens: List[str],
    output_path: str,
    select_layers: List[int] = None,
    title: str = None,
):
    """
    Show attention patterns for a few selected layers (not all 32).
    Typically 4-6 layers are shown in a grid.
    """
    num_layers = len(attentions)

    if select_layers is None:
        # Pick 6 representative layers
        if num_layers <= 6:
            select_layers = list(range(num_layers))
        else:
            select_layers = [0, num_layers//4, num_layers//2, 
                           3*num_layers//4, num_layers - 2, num_layers - 1]

    n_plots = len(select_layers)
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols

    display_tokens = [_clean_token_label(t) for t in tokens]

    fig_h = min(DOUBLE_COL_WIDTH * rows / cols * 0.75, MAX_HEIGHT)
    fig, axes = plt.subplots(rows, cols, figsize=(DOUBLE_COL_WIDTH, fig_h))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, layer_idx in enumerate(select_layers):
        avg_attn = attentions[layer_idx][0].mean(dim=0).float().cpu().numpy()
        ax = axes[i]
        im = ax.imshow(avg_attn, cmap='YlOrRd', aspect='equal', interpolation='nearest')
        ax.set_title(f'Layer {layer_idx}', fontsize=BODY_FS, pad=3)

        # Only show ticks on edge plots to save space. Small multi-panel
        # layout → tick labels stay at TINY_FS (dense rotated tokens).
        if i >= (rows - 1) * cols:
            n_show = min(len(display_tokens), 15)
            tick_positions = np.linspace(0, len(display_tokens)-1, n_show, dtype=int)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([display_tokens[j] for j in tick_positions],
                               rotation=90, ha='center', fontsize=TINY_FS)
        else:
            ax.set_xticks([])

        if i % cols == 0:
            n_show = min(len(display_tokens), 15)
            tick_positions = np.linspace(0, len(display_tokens)-1, n_show, dtype=int)
            ax.set_yticks(tick_positions)
            ax.set_yticklabels([display_tokens[j] for j in tick_positions],
                               fontsize=TINY_FS)
        else:
            ax.set_yticks([])

    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    if title:
        fig.suptitle(title, fontsize=BODY_FS, y=1.01)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.012, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Attention', fontsize=BODY_FS)
    cbar.ax.tick_params(labelsize=SMALL_FS, width=0.4)
    cbar.outline.set_linewidth(0.5)

    _save_figure(fig, output_path)


# ============================================================================
# Figure 5: Selected Head Attention in One Layer (2×2 grid)
# ============================================================================
def plot_selected_head_patterns(
    attentions: Tuple[torch.Tensor],
    tokens: List[str],
    layer: int,
    output_path: str,
    select_heads: List[int] = None,
    title: str = None,
):
    """Show attention for selected heads in a specific layer (4-6 heads)."""
    attn = attentions[layer][0]  # (num_heads, seq_len, seq_len)
    num_heads = attn.shape[0]

    if select_heads is None:
        # Pick 4 representative heads
        if num_heads <= 4:
            select_heads = list(range(num_heads))
        else:
            select_heads = [0, num_heads//3, 2*num_heads//3, num_heads - 1]

    n_plots = len(select_heads)
    cols = min(2, n_plots)
    rows = (n_plots + cols - 1) // cols

    display_tokens = [_clean_token_label(t) for t in tokens]

    fig_h = min(DOUBLE_COL_WIDTH * rows / cols * 0.8, MAX_HEIGHT)
    fig, axes = plt.subplots(rows, cols, figsize=(DOUBLE_COL_WIDTH, fig_h))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, head_idx in enumerate(select_heads):
        head_attn = attn[head_idx].float().cpu().numpy()
        ax = axes[i]
        im = ax.imshow(head_attn, cmap='YlOrRd', aspect='equal', interpolation='nearest')
        ax.set_title(f'Head {head_idx}', fontsize=BODY_FS, pad=3)

        if i >= (rows - 1) * cols:
            n_show = min(len(display_tokens), 12)
            tick_positions = np.linspace(0, len(display_tokens)-1, n_show, dtype=int)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([display_tokens[j] for j in tick_positions],
                               rotation=90, ha='center', fontsize=TINY_FS)
        else:
            ax.set_xticks([])

        if i % cols == 0:
            n_show = min(len(display_tokens), 12)
            tick_positions = np.linspace(0, len(display_tokens)-1, n_show, dtype=int)
            ax.set_yticks(tick_positions)
            ax.set_yticklabels([display_tokens[j] for j in tick_positions],
                               fontsize=TINY_FS)
        else:
            ax.set_yticks([])

    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    if title:
        fig.suptitle(title, fontsize=BODY_FS, y=1.01)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.012, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Attention', fontsize=BODY_FS)
    cbar.ax.tick_params(labelsize=SMALL_FS, width=0.4)
    cbar.outline.set_linewidth(0.5)

    _save_figure(fig, output_path)


# ============================================================================
# Figure 6: SMILES Feature Attention Analysis
# ============================================================================
# ...existing code...
def plot_smiles_feature_attention(
    attentions: Tuple[torch.Tensor],
    tokens: List[str],
    output_path: str,
    title: str = None,
) -> Dict:
    """SMILES feature category attention analysis with publication quality.
    
    Uses column-wise mean (consistent with all other panels).
    """
    all_attn = torch.stack([attn[0] for attn in attentions])
    avg_attn_matrix = all_attn.mean(dim=(0, 1)).float().cpu().numpy()  # (seq, seq)
    token_attention = avg_attn_matrix.mean(axis=0)  # column-wise mean -> (seq,)

    categories = _categorize_tokens(tokens, token_attention)

    stats = {}
    for cat, values in categories.items():
        if values:
            stats[cat] = {'mean': np.mean(values), 'std': np.std(values), 'count': len(values)}

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, SINGLE_COL_WIDTH * 0.7))

    cat_names = list(stats.keys())
    cat_means = [stats[c]['mean'] for c in cat_names]
    cat_stds = [stats[c]['std'] for c in cat_names]

    x = np.arange(len(cat_names))
    # ACS-friendly color palette (colorblind-safe, up to 8 categories)
    colors_list = ['#d73027', '#fc8d59', '#fee08b', '#91bfdb', '#4575b4',
                   '#762a83', '#1b7837', '#b15928']
    bars = ax.bar(x, cat_means, yerr=cat_stds, capsize=2.5, 
                  color=colors_list[:len(cat_names)], alpha=0.85,
                  edgecolor='black', linewidth=0.5, width=0.6,
                  error_kw={'linewidth': 0.6, 'capthick': 0.6})

    ax.set_xticks(x)
    ax.set_xticklabels(cat_names, fontsize=BODY_FS, rotation=20, ha='right')
    ax.set_ylabel('Avg Attn Received', fontsize=BODY_FS)
    ax.tick_params(axis='y', labelsize=BODY_FS)
    if title:
        ax.set_title(title, fontsize=BODY_FS, pad=6)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Bar count labels: small white text on dark bbox. n labels are short
    # (n=NN), so SMALL_FS fits inside the bar without overlap.
    for i, bar in enumerate(bars):
        bar_height = bar.get_height()
        label_y = bar_height * 0.4
        ax.text(bar.get_x() + bar.get_width() / 2., label_y,
                f'n={int(stats[cat_names[i]]["count"])}',
                ha='center', va='center', fontsize=SMALL_FS,
                color='white',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='black',
                          alpha=0.5, linewidth=0))
    plt.tight_layout()
    _save_figure(fig, output_path)

    return stats

# ============================================================================
# Figure 7: Canonical vs Isomeric Comparison
# ============================================================================
# ...existing code...
def plot_canonical_vs_isomeric(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    canonical_smiles: str,
    isomeric_smiles: str,
    output_path: str,
    device: str = "cpu",
):
    """Side-by-side comparison of canonical vs isomeric attention.
    
    Uses column-wise mean (consistent with all other panels).
    """
    canonical_attn, canonical_tokens, _, _ = get_attention_weights(model, tokenizer, canonical_smiles, device)
    isomeric_attn, isomeric_tokens, _, _ = get_attention_weights(model, tokenizer, isomeric_smiles, device)

    canonical_avg = torch.stack([attn[0] for attn in canonical_attn]).mean(dim=(0, 1))
    isomeric_avg = torch.stack([attn[0] for attn in isomeric_attn]).mean(dim=(0, 1))

    # Column-wise mean (how much each token receives)
    c_importance = canonical_avg.float().cpu().numpy().mean(axis=0)
    i_importance = isomeric_avg.float().cpu().numpy().mean(axis=0)

    # Normalize
    c_norm = c_importance / c_importance.max()
    i_norm = i_importance / i_importance.max()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.45))

    c_tokens_clean = [_clean_token_label(t) for t in canonical_tokens]
    i_tokens_clean = [_clean_token_label(t) for t in isomeric_tokens]

    # Many rotated tokens → use SMALL_FS only when density demands it
    c_tick_fs = SMALL_FS if len(c_tokens_clean) > 25 else BODY_FS
    i_tick_fs = SMALL_FS if len(i_tokens_clean) > 25 else BODY_FS

    c_colors = plt.cm.YlOrRd(c_norm)
    ax1.bar(range(len(c_tokens_clean)), c_importance, color=c_colors,
            edgecolor='grey', linewidth=0.5, width=0.8)
    ax1.set_xticks(range(len(c_tokens_clean)))
    ax1.set_xticklabels(c_tokens_clean, rotation=90, ha='center', fontsize=c_tick_fs)
    ax1.set_ylabel('Avg Attn Received', fontsize=BODY_FS)
    ax1.tick_params(axis='y', labelsize=BODY_FS)
    ax1.set_title('a', fontsize=PANEL_FS, loc='left', pad=4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    i_colors = plt.cm.YlOrRd(i_norm)
    ax2.bar(range(len(i_tokens_clean)), i_importance, color=i_colors,
            edgecolor='grey', linewidth=0.5, width=0.8)
    ax2.set_xticks(range(len(i_tokens_clean)))
    ax2.set_xticklabels(i_tokens_clean, rotation=90, ha='center', fontsize=i_tick_fs)
    ax2.set_ylabel('Avg Attn Received', fontsize=BODY_FS)
    ax2.tick_params(axis='y', labelsize=BODY_FS)
    ax2.set_title('b', fontsize=BODY_FS, loc='left', pad=4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout(h_pad=1.2)
    _save_figure(fig, output_path)

    # Entropy analysis
    def attention_entropy(attn):
        attn = attn.float().cpu().numpy()
        attn = attn / attn.sum(axis=-1, keepdims=True)
        attn = np.clip(attn, 1e-10, 1)
        entropy = -np.sum(attn * np.log(attn), axis=-1)
        return entropy.mean()

    return {
        'canonical_entropy': float(attention_entropy(canonical_avg)),
        'isomeric_entropy': float(attention_entropy(isomeric_avg)),
    }

# ============================================================================
# DTW Analysis: Quantitative Comparison of Canonical vs Isomeric Attention
# ============================================================================
def _dtw_distance_and_alignment(seq1: np.ndarray, seq2: np.ndarray):
    """
    Compute Dynamic Time Warping distance and optimal alignment path.
    
    Returns:
        dtw_dist: float, the DTW distance (unnormalized)
        dtw_dist_norm: float, DTW distance normalized by path length
        path: list of (i, j) tuples, the optimal alignment
    """
    n, m = len(seq1), len(seq2)
    # Cost matrix
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = (seq1[i-1] - seq2[j-1]) ** 2
            cost[i, j] = d + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
    
    # Backtrack to find optimal path
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        candidates = [
            (cost[i-1, j-1], i-1, j-1),
            (cost[i-1, j], i-1, j),
            (cost[i, j-1], i, j-1),
        ]
        _, i, j = min(candidates, key=lambda x: x[0])
    path.reverse()
    
    dtw_dist = np.sqrt(cost[n, m])
    dtw_dist_norm = dtw_dist / len(path)
    
    return dtw_dist, dtw_dist_norm, path

# ...existing code...
def compute_dtw_analysis(
    canonical_importance: np.ndarray,
    isomeric_importance: np.ndarray,
    canonical_tokens: List[str],
    isomeric_tokens: List[str],
    n_permutations: int = 10000,
) -> Dict:
    """
    Comprehensive DTW-based comparison of canonical vs isomeric attention.
    
    Computes:
    1. DTW distance (raw and normalized)
    2. Pearson/Spearman correlation after DTW alignment
    3. Random baseline correlation (permutation test, default 10,000 iterations)
    4. Normalized attention curves used for comparison
    """
    # Normalize to [0, 1]
    c_norm = canonical_importance / canonical_importance.max()
    i_norm = isomeric_importance / isomeric_importance.max()
    
    # 1. DTW distance and alignment
    dtw_dist, dtw_dist_norm, path = _dtw_distance_and_alignment(c_norm, i_norm)
    
    # 2. Extract aligned sequences
    c_aligned = np.array([c_norm[p[0]] for p in path])
    i_aligned = np.array([i_norm[p[1]] for p in path])
    
    # 3. Pearson and Spearman correlation on aligned sequences
    pearson_r, pearson_p = pearsonr(c_aligned, i_aligned)
    spearman_r, spearman_p = spearmanr(c_aligned, i_aligned)
    
    # 4. Random baseline: permutation test
    if n_permutations > 0:
        random_pearson_rs = []
        rng = np.random.RandomState(42)
        for _ in range(n_permutations):
            shuffled = rng.permutation(i_norm)
            _, _, rand_path = _dtw_distance_and_alignment(c_norm, shuffled)
            c_rand_aligned = np.array([c_norm[p[0]] for p in rand_path])
            i_rand_aligned = np.array([shuffled[p[1]] for p in rand_path])
            r_rand, _ = pearsonr(c_rand_aligned, i_rand_aligned)
            random_pearson_rs.append(r_rand)

        random_baseline_mean = np.mean(random_pearson_rs)
        random_baseline_std = np.std(random_pearson_rs)
        # p-value: fraction of random correlations >= observed
        p_value_permutation = np.mean(np.array(random_pearson_rs) >= pearson_r)
    else:
        random_baseline_mean = float('nan')
        random_baseline_std = float('nan')
        p_value_permutation = float('nan')
    
    return {
        'dtw_distance': float(dtw_dist),
        'dtw_distance_normalized': float(dtw_dist_norm),
        'path_length': len(path),
        'pearson_r_aligned': float(pearson_r),
        'pearson_p_aligned': float(pearson_p),
        'spearman_r_aligned': float(spearman_r),
        'spearman_p_aligned': float(spearman_p),
        'random_baseline_pearson_mean': float(random_baseline_mean),
        'random_baseline_pearson_std': float(random_baseline_std),
        'p_value_permutation': float(p_value_permutation),
        'n_permutations': n_permutations,
        'canonical_tokens_count': len(canonical_importance),
        'isomeric_tokens_count': len(isomeric_importance),
        'canonical_norm_max': float(c_norm.max()),
        'isomeric_norm_max': float(i_norm.max()),
        'alignment_path': [(int(p[0]), int(p[1])) for p in path],
    }

def compute_feature_level_correlation(
    canonical_importance: np.ndarray,
    isomeric_importance: np.ndarray,
    canonical_tokens: List[str],
    isomeric_tokens: List[str],
) -> Dict:
    """
    Compute correlation at the chemical feature category level.
    Aggregates token attention by SMILES feature type, then correlates.
    """
    c_features_raw = _categorize_tokens(
        canonical_tokens, canonical_importance / canonical_importance.max())
    i_features_raw = _categorize_tokens(
        isomeric_tokens, isomeric_importance / isomeric_importance.max())
    
    c_features = {k: np.mean(v) if v else 0.0 for k, v in c_features_raw.items()}
    i_features = {k: np.mean(v) if v else 0.0 for k, v in i_features_raw.items()}
    
    # Ensure same category order (union of both)
    cats = sorted(set(c_features.keys()) | set(i_features.keys()))
    c_vals = np.array([c_features.get(c, 0.0) for c in cats])
    i_vals = np.array([i_features.get(c, 0.0) for c in cats])
    
    if len(cats) >= 3:
        pearson_r, pearson_p = pearsonr(c_vals, i_vals)
        spearman_r, spearman_p = spearmanr(c_vals, i_vals)
    else:
        pearson_r = pearson_p = spearman_r = spearman_p = float('nan')
    
    return {
        'categories': cats,
        'canonical_feature_attention': {c: float(c_features.get(c, 0.0)) for c in cats},
        'isomeric_feature_attention': {c: float(i_features.get(c, 0.0)) for c in cats},
        'feature_pearson_r': float(pearson_r),
        'feature_pearson_p': float(pearson_p),
        'feature_spearman_r': float(spearman_r),
        'feature_spearman_p': float(spearman_p),
    }

# ============================================================================
# Cross-Molecule Control: Representation Invariance Validation
# ============================================================================
def compute_cross_molecule_control(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    canonical_input: str,
    control_inputs: List[str],
    device: str = "cpu",
) -> Dict:
    """
    Cross-molecule DTW control for representation invariance validation.

    Computes DTW correlation between the canonical molecule and each control
    molecule.  If same-molecule r (canonical vs isomeric) >> cross-molecule r,
    the high same-molecule correlation cannot be attributed to universal MOF
    SMILES properties and instead reflects genuine representation invariance.
    """
    can_attn, can_tokens, _, _ = get_attention_weights(
        model, tokenizer, canonical_input, device)
    can_avg = torch.stack([attn[0] for attn in can_attn]).mean(dim=(0, 1))
    can_importance = can_avg.float().cpu().numpy().mean(axis=0)

    # Per-zone feature breakdown for canonical (3 zones)
    can_n_layers = len(can_attn)
    can_full_attn = np.zeros((can_n_layers, len(can_tokens)))
    for li in range(can_n_layers):
        la = can_attn[li][0].mean(dim=0).float().cpu().numpy()
        can_full_attn[li] = la.mean(axis=0)

    zone_ranges = [(0, 10), (10, 20), (20, can_n_layers)]
    zone_keys = ['L0-10_local', 'L11-20_functional', f'L21-{can_n_layers-1}_topology']
    key_cats = ['Ring Closures', 'Branches', 'Bonds', 'Aromatic', 'Metal', 'MOFid', 'Atoms']

    comparisons = []
    for ctrl_idx, ctrl_input in enumerate(control_inputs):
        print(f"  Cross-molecule DTW [{ctrl_idx+1}/{len(control_inputs)}]: "
              f"{ctrl_input[:70]}...")
        ctrl_attn, ctrl_tokens, _, _ = get_attention_weights(
            model, tokenizer, ctrl_input, device)
        ctrl_avg = torch.stack([attn[0] for attn in ctrl_attn]).mean(dim=(0, 1))
        ctrl_importance = ctrl_avg.float().cpu().numpy().mean(axis=0)

        dtw_res = compute_dtw_analysis(
            can_importance, ctrl_importance, can_tokens, ctrl_tokens,
            n_permutations=0)
        feat_res = compute_feature_level_correlation(
            can_importance, ctrl_importance, can_tokens, ctrl_tokens)

        # Per-zone feature breakdown for control
        ctrl_n_layers = len(ctrl_attn)
        ctrl_full_attn = np.zeros((ctrl_n_layers, len(ctrl_tokens)))
        for li in range(ctrl_n_layers):
            la = ctrl_attn[li][0].mean(dim=0).float().cpu().numpy()
            ctrl_full_attn[li] = la.mean(axis=0)

        ctrl_zone_features = {}
        for zk, (zs, ze) in zip(zone_keys, zone_ranges):
            ze_c = min(ze, ctrl_n_layers)
            zone_imp = ctrl_full_attn[zs:ze_c].mean(axis=0)
            zone_cats = _categorize_tokens(ctrl_tokens, zone_imp,
                                           categories_order=key_cats)
            ctrl_zone_features[zk] = {
                k: float(np.mean(v)) if v else 0.0
                for k, v in zone_cats.items()
            }

        comparisons.append({
            'control_input': ctrl_input,
            'control_seq_length': len(ctrl_tokens),
            'dtw_pearson_r': dtw_res['pearson_r_aligned'],
            'dtw_spearman_r': dtw_res['spearman_r_aligned'],
            'dtw_distance': dtw_res['dtw_distance'],
            'dtw_distance_normalized': dtw_res['dtw_distance_normalized'],
            'path_length': dtw_res['path_length'],
            'feature_pearson_r': feat_res['feature_pearson_r'],
            'feature_spearman_r': feat_res['feature_spearman_r'],
            'canonical_feature_attn': feat_res['canonical_feature_attention'],
            'control_feature_attn': feat_res['isomeric_feature_attention'],
            'categories': feat_res['categories'],
            'control_per_zone_features': ctrl_zone_features,
        })

    # Aggregate cross-molecule statistics
    dtw_rs = [c['dtw_pearson_r'] for c in comparisons]
    feat_rs = [c['feature_pearson_r'] for c in comparisons]

    return {
        'canonical_input': canonical_input,
        'canonical_seq_length': len(can_tokens),
        'n_controls': len(comparisons),
        'comparisons': comparisons,
        'cross_molecule_dtw_r_mean': float(np.mean(dtw_rs)) if dtw_rs else float('nan'),
        'cross_molecule_dtw_r_std': float(np.std(dtw_rs)) if dtw_rs else float('nan'),
        'cross_molecule_feat_r_mean': float(np.mean(feat_rs)) if feat_rs else float('nan'),
        'cross_molecule_feat_r_std': float(np.std(feat_rs)) if feat_rs else float('nan'),
    }


def plot_invariance_control_comparison(
    same_mol_dtw_r: float,
    same_mol_feat_r: float,
    cross_mol_results: Dict,
    random_baseline_r: float,
    random_baseline_std: float,
    output_path: str,
):
    """
    Publication-quality figure comparing same-molecule vs cross-molecule
    DTW and feature-level correlations, with random baseline.

    Two-panel bar chart:
      (a) Token-level DTW Pearson r
      (b) Category-level Pearson r
    """
    comparisons = cross_mol_results['comparisons']
    n_ctrl = len(comparisons)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.35))

    # ---- Panel (a): Token-level DTW Pearson r ----
    labels_a, vals_a, colors_a = [], [], []

    labels_a.append('Same mol.\n(Can. vs Iso.)')
    vals_a.append(same_mol_dtw_r)
    colors_a.append('#d73027')

    for i, comp in enumerate(comparisons):
        labels_a.append(f'Control {i+1}')
        vals_a.append(comp['dtw_pearson_r'])
        colors_a.append('#4575b4')

    labels_a.append('Random\nbaseline')
    vals_a.append(random_baseline_r)
    colors_a.append('#999999')

    x_a = np.arange(len(labels_a))
    bars_a = ax1.bar(x_a, vals_a, color=colors_a, edgecolor='black',
                     linewidth=0.5, width=0.6, alpha=0.85)
    ax1.errorbar(x_a[-1], random_baseline_r, yerr=random_baseline_std,
                 fmt='none', ecolor='black', capsize=3, linewidth=0.8)
    ax1.set_xticks(x_a)
    # Multi-line category labels — SMALL_FS avoids x-axis crowding
    ax1.set_xticklabels(labels_a, fontsize=SMALL_FS)
    ax1.set_ylabel('Pearson $r$ (DTW-aligned)', fontsize=BODY_FS)
    ax1.tick_params(axis='y', labelsize=BODY_FS)
    ax1.set_title('a', fontsize=PANEL_FS, loc='left', pad=4)
    ax1.set_ylim(0, 1.25)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    for bar, val in zip(bars_a, vals_a):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=SMALL_FS)

    # ---- Panel (b): Category-level Pearson r ----
    labels_b, vals_b, colors_b = [], [], []

    labels_b.append('Same mol.\n(Can. vs Iso.)')
    vals_b.append(same_mol_feat_r)
    colors_b.append('#d73027')

    for i, comp in enumerate(comparisons):
        labels_b.append(f'Control {i+1}')
        vals_b.append(comp['feature_pearson_r'])
        colors_b.append('#4575b4')

    x_b = np.arange(len(labels_b))
    bars_b = ax2.bar(x_b, vals_b, color=colors_b, edgecolor='black',
                     linewidth=0.5, width=0.6, alpha=0.85)
    ax2.set_xticks(x_b)
    ax2.set_xticklabels(labels_b, fontsize=SMALL_FS)
    ax2.set_ylabel('Pearson $r$ (feature-level)', fontsize=BODY_FS)
    ax2.tick_params(axis='y', labelsize=BODY_FS)
    ax2.set_title('b', fontsize=BODY_FS, loc='left', pad=4)
    ax2.set_ylim(0, 1.25)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    for bar, val in zip(bars_b, vals_b):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=SMALL_FS)

    plt.tight_layout(w_pad=2.0)
    _save_figure(fig, output_path)


# ============================================================================
# Figure 9: Hierarchical Attention Overview (Inspired by reference heatmap)
# ============================================================================
def plot_hierarchical_attention_overview(
    attentions: Tuple[torch.Tensor],
    tokens: List[str],
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    canonical_input: str,
    isomeric_input: str,
    output_path: str,
    device: str = "cpu",
    deep_start: int = 24,
    num_layers: int = 32,
):
    """
    Publication-quality hierarchical attention overview figure.
    """
    import matplotlib.gridspec as gridspec

    display_tokens = [_clean_token_label(t) for t in tokens]
    n_tokens = len(display_tokens)
    n_layers = len(attentions)

    # 1. Compute full-layer attention matrix
    full_attn_matrix = np.zeros((n_layers, n_tokens))
    for li in range(n_layers):
        layer_attn = attentions[li][0].mean(dim=0).float().cpu().numpy()
        full_attn_matrix[li] = layer_attn.mean(axis=0)

    # Row-wise normalization
    row_min = full_attn_matrix.min(axis=1, keepdims=True)
    row_max = full_attn_matrix.max(axis=1, keepdims=True)
    row_range = row_max - row_min
    row_range[row_range == 0] = 1.0
    attn_norm = (full_attn_matrix - row_min) / row_range

    # 2. Token classification for annotation
    token_classes = [_classify_smiles_token(t) for t in tokens]
    metal_indices = [i for i, c in enumerate(token_classes) if c == 'Metal']
    mofid_indices = [i for i, c in enumerate(token_classes) if c == 'MOFid']
    func_group_indices = []
    for i, t in enumerate(tokens):
        clean = t.replace('▁', '').replace('Ġ', '')
        if clean.startswith('[') and any(ch in clean for ch in ['-', '+']):
            func_group_indices.append(i)
        elif clean in ('C(=O)', 'C(', '=O', '=') and token_classes[i] in ('Atoms', 'Bonds'):
            pass

    # =========================================================================
    # 3. Build figure with RESTRUCTURED GridSpec.
    #    The left column has 4 rows:
    #        row 0 = annotation strip ABOVE heatmap (Metal / MOFid callouts
    #                live here so they never overlap heatmap cells),
    #        row 1 = heatmap + zone-annotation bar,
    #        row 2 = buffer for rotated x-tick labels,
    #        row 3 = horizontal colorbar.
    # =========================================================================
    fig = plt.figure(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.82))

    gs_main = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[6.5, 3.5],
                                wspace=0.28)

    gs_left = gridspec.GridSpecFromSubplotSpec(
        4, 2, subplot_spec=gs_main[0, 0],
        width_ratios=[2.5, 30],
        # row heights: annotation strip / heatmap / x-label buffer / colorbar
        height_ratios=[0.12, 1.0, 0.22, 0.03],
        wspace=0.06,
        hspace=0.05,
    )

    gs_right = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs_main[0, 1],
        height_ratios=[1, 1, 1], hspace=0.55,
    )

    ax_heat = fig.add_subplot(gs_left[1, 1])
    ax_bar = fig.add_subplot(gs_left[1, 0], sharey=ax_heat)
    ax_cbar = fig.add_subplot(gs_left[3, 1])

    ax_b = fig.add_subplot(gs_right[0])
    ax_c = fig.add_subplot(gs_right[1])
    ax_d = fig.add_subplot(gs_right[2])

    # =========================================================================
    # 4. Panel (a): Layer-Zone Annotation Bar & Main Heatmap
    # =========================================================================
    zones = [
        (0, 10, 'Local Chemical\nBonds', '#2c3e50'),
        (10, 20, 'Functional Group\nRecognition', '#7f8c8d'),
        (20, n_layers, 'Global\nTopology', '#bdc3c7'),
    ]

    attn_display = attn_norm[::-1, :]
    im = ax_heat.imshow(
        attn_display, cmap='YlOrRd', aspect='auto',
        interpolation='nearest', vmin=0, vmax=1
    )

    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylabel('Transformer Layer', fontsize=BODY_FS, labelpad=6)

    for zone_start, zone_end, zone_label, zone_color in zones:
        y_bottom = n_layers - zone_end - 0.5
        height = zone_end - zone_start
        rect = plt.Rectangle(
            (0, y_bottom), 1, height,
            facecolor=zone_color, edgecolor='black', linewidth=0.5
        )
        ax_bar.add_patch(rect)
        y_mid = y_bottom + height / 2.0
        text_color = 'white' if zone_color in ('#2c3e50',) else 'black'
        # Zone strip is narrow; SMALL_FS is the largest size that fits rotated
        ax_bar.text(0.5, y_mid, zone_label, ha='center', va='center',
                    fontsize=SMALL_FS, color=text_color,
                    rotation=90)

    ax_bar.set_xticks([])
    for spine in ax_bar.spines.values():
        spine.set_visible(False)

    ax_heat.set_title('a', fontsize=PANEL_FS, loc='left', pad=4)
    ax_heat.set_xticks(np.arange(n_tokens))
    # ~20 tokens rotated 90°; SMALL_FS keeps them readable without overlap
    ax_heat.set_xticklabels(display_tokens, rotation=90, ha='center',
                            fontsize=SMALL_FS)
    ax_heat.set_xlabel('Token Position', fontsize=BODY_FS, labelpad=6)

    layer_labels_all = list(range(n_layers))[::-1]
    y_ticks = np.arange(n_layers)
    y_labels = [str(layer_labels_all[i]) if i % 3 == 0 else '' for i in range(n_layers)]
    ax_heat.set_yticks(y_ticks)
    # Many layer ticks stacked vertically; SMALL_FS fits without overlap
    ax_heat.set_yticklabels(y_labels, fontsize=SMALL_FS)

    ax_heat.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax_bar.tick_params(axis='y', which='both', left=True, labelleft=True,
                       labelsize=SMALL_FS)

    for zone_start, zone_end, _, _ in zones[1:]:
        y_line = n_layers - zone_start - 0.5
        ax_heat.axhline(y=y_line, color='white', linewidth=1.2,
                        linestyle='--', alpha=0.8)

    # ---- Callouts for key token groups ----
    # Annotations are drawn OUTSIDE the heatmap (above it) with
    # annotation_clip=False, so the text never overlaps heatmap cells.
    bbox_style = dict(boxstyle='round,pad=0.25', facecolor='white',
                      alpha=0.95, edgecolor='none')

    # The heatmap's top edge in data coords is y = -0.5; put text labels at
    # y = -3 (outside the axes, inside the reserved "annotation strip" row).
    label_y = -3.0

    # Metal-vs-MOFid callouts sit side by side in the annotation strip above
    # the heatmap. When metal and MOFid tokens are adjacent (e.g. "[Zn][Zn] ."
    # → "MOFid-v1"), the two labels collide horizontally. Fix by anchoring
    # them on opposite sides of their arrow tips (right-aligned / left-aligned)
    # and nudging the text position a little outward.
    if metal_indices:
        m_start = min(metal_indices) - 0.5
        m_end = max(metal_indices) + 0.5
        row_top = -0.5
        row_bottom = n_layers - 1 - 20 + 0.5
        rect = plt.Rectangle((m_start, row_top), m_end - m_start,
                             row_bottom - row_top,
                             fill=False, edgecolor='#c0392b',
                             linewidth=1.2, linestyle='-')
        ax_heat.add_patch(rect)
        metal_anchor_x = np.mean(metal_indices)
        ax_heat.annotate(
            'Metal Center',
            xy=(metal_anchor_x, row_top),
            xytext=(metal_anchor_x - 0.6, label_y),
            fontsize=SMALL_FS, color='#c0392b', bbox=bbox_style,
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=0.8,
                            connectionstyle='arc3,rad=0.0'),
            ha='right', va='bottom', annotation_clip=False,
        )

    if mofid_indices:
        mof_start = min(mofid_indices) - 0.5
        mof_end = max(mofid_indices) + 0.5
        row_top = -0.5
        row_bottom = n_layers - 1 - 20 + 0.5
        rect = plt.Rectangle((mof_start, row_top), mof_end - mof_start,
                             row_bottom - row_top,
                             fill=False, edgecolor='#2980b9',
                             linewidth=1.2, linestyle='-')
        ax_heat.add_patch(rect)
        mofid_anchor_x = np.mean(mofid_indices)
        ax_heat.annotate(
            'MOFid Topology',
            xy=(mofid_anchor_x, row_top),
            xytext=(mofid_anchor_x + 0.6, label_y),
            fontsize=SMALL_FS, color='#2980b9', bbox=bbox_style,
            arrowprops=dict(arrowstyle='->', color='#2980b9', lw=0.8,
                            connectionstyle='arc3,rad=0.0'),
            ha='left', va='bottom', annotation_clip=False,
        )

    if func_group_indices:
        fg_start = min(func_group_indices) - 0.5
        fg_end = max(func_group_indices) + 0.5
        row_top_fg = n_layers - 1 - 19 - 0.5
        row_bottom_fg = n_layers - 1 - 10 + 0.5
        rect = plt.Rectangle((fg_start, row_top_fg), fg_end - fg_start,
                             row_bottom_fg - row_top_fg,
                             fill=False, edgecolor='#c0392b',
                             linewidth=1.0, linestyle='--')
        ax_heat.add_patch(rect)
        ax_heat.annotate(
            'Functional Groups',
            xy=(np.mean(func_group_indices), row_top_fg),
            xytext=(np.mean(func_group_indices), label_y),
            fontsize=SMALL_FS, color='#c0392b', bbox=bbox_style,
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=0.8),
            ha='center', va='bottom', annotation_clip=False,
        )

    cbar = fig.colorbar(im, cax=ax_cbar, orientation='horizontal')
    cbar.set_label('Normalized Attention (per-layer)', fontsize=BODY_FS)
    cbar.ax.tick_params(labelsize=SMALL_FS, width=0.4)
    cbar.outline.set_linewidth(0.4)
    cbar.ax.text(0.0, -3.0, 'Low', transform=cbar.ax.transAxes,
                 fontsize=SMALL_FS, ha='left', va='top', color='grey')
    cbar.ax.text(1.0, -3.0, 'High', transform=cbar.ax.transAxes,
                 fontsize=SMALL_FS, ha='right', va='top', color='grey')


    # =========================================================================
    # 6. Panel (b): Feature-Category Bar Chart
    # =========================================================================
    deep_attn_matrix = full_attn_matrix[deep_start:]
    deep_importance = deep_attn_matrix.mean(axis=0)  

    categories = _categorize_tokens(tokens, deep_importance)
    stats_b = {}
    for cat, values in categories.items():
        if values:
            stats_b[cat] = {'mean': np.mean(values), 'std': np.std(values), 'count': len(values)}

    cat_names = list(stats_b.keys())
    cat_means = [stats_b[c]['mean'] for c in cat_names]
    cat_stds = [stats_b[c]['std'] for c in cat_names]
    n_cats = len(cat_names)
    xb = np.arange(n_cats)
    colors_b = ['#d73027', '#fc8d59', '#fee08b', '#91bfdb', '#4575b4', '#762a83', '#1b7837', '#b15928']

    bars = ax_b.barh(xb, cat_means, xerr=cat_stds, capsize=1.5,
                     color=colors_b[:n_cats], alpha=0.85,
                     edgecolor='black', linewidth=0.4, height=0.6,
                     error_kw={'linewidth': 0.5, 'capthick': 0.5})
    ax_b.set_yticks(xb)
    # Right column is narrow (~2.5"); SMALL_FS keeps category names readable
    ax_b.set_yticklabels(cat_names, fontsize=SMALL_FS)
    ax_b.set_xlabel('Avg Attn Received', fontsize=SMALL_FS)
    ax_b.set_title('b', fontsize=BODY_FS, loc='left', pad=4)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.tick_params(axis='x', labelsize=SMALL_FS)
    ax_b.invert_yaxis()
    for i, bar in enumerate(bars):
        w = bar.get_width()
        ax_b.text(w * 0.5, bar.get_y() + bar.get_height() / 2,
                  f'n={int(stats_b[cat_names[i]]["count"])}',
                  ha='center', va='center', fontsize=TINY_FS,
                  color='white' if w > max(cat_means) * 0.3 else 'black')

    # =========================================================================
    # 7. Panel (c): Canonical vs Isomeric DTW Comparison
    # =========================================================================
    canonical_attn, canonical_tokens_raw, _, _ = get_attention_weights(model, tokenizer, canonical_input, device)
    isomeric_attn, isomeric_tokens_raw, _, _ = get_attention_weights(model, tokenizer, isomeric_input, device)

    c_avg = torch.stack([attn[0] for attn in canonical_attn]).mean(dim=(0, 1))
    i_avg = torch.stack([attn[0] for attn in isomeric_attn]).mean(dim=(0, 1))
    c_importance = c_avg.float().cpu().numpy().mean(axis=0)
    i_importance = i_avg.float().cpu().numpy().mean(axis=0)
    c_norm_c = c_importance / c_importance.max()
    i_norm_c = i_importance / i_importance.max()

    ax_c.plot(np.arange(len(c_norm_c)), c_norm_c, color='#d73027', linewidth=0.8,
              marker='o', markersize=2, label=f'Canonical ({len(c_norm_c)})', alpha=0.85)
    ax_c.plot(np.arange(len(i_norm_c)), i_norm_c, color='#4575b4', linewidth=0.8,
              marker='s', markersize=2, label=f'Isomeric ({len(i_norm_c)})', alpha=0.85)
    ax_c.set_xlabel('Token Position', fontsize=SMALL_FS)
    ax_c.set_ylabel('Norm. Attn', fontsize=SMALL_FS)
    ax_c.set_title('c', fontsize=BODY_FS, loc='left', pad=4)
    ax_c.legend(fontsize=SMALL_FS, loc='upper right', framealpha=0.8)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.tick_params(axis='both', labelsize=SMALL_FS)
    ax_c.set_ylim(-0.05, 1.20)

    dtw_results = compute_dtw_analysis(c_importance, i_importance, canonical_tokens_raw, isomeric_tokens_raw)
    feature_results = compute_feature_level_correlation(c_importance, i_importance, canonical_tokens_raw, isomeric_tokens_raw)

    # Small annotation box in bottom-left corner; TINY_FS because this
    # narrow panel cannot fit two lines of body-size text alongside the plot
    dtw_text = (f"DTW $r$ = {dtw_results['pearson_r_aligned']:.2f}\n"
                f"Random = {dtw_results['random_baseline_pearson_mean']:.2f}"
                f"\u00b1{dtw_results['random_baseline_pearson_std']:.2f}")
    ax_c.text(0.03, 0.05, dtw_text, transform=ax_c.transAxes, fontsize=TINY_FS,
              verticalalignment='bottom',
              bbox=dict(boxstyle='round,pad=0.25', facecolor='wheat', alpha=0.7))

    # =========================================================================
    # 8. Panel (d): Layer-Zone Attention Summary
    # =========================================================================
    zone_names = ['L0\u201310\n(Local)', 'L11\u201320\n(Func.Grp)', f'L21\u2013{n_layers-1}\n(Topology)']
    zone_ranges = [(0, 10), (10, 20), (20, n_layers)]

    key_categories_all = ['Branches', 'Aromatic', 'Bonds', 'Metal', 'MOFid', 'Atoms']
    zone_cat_data = {}
    for zname, (zs, ze) in zip(zone_names, zone_ranges):
        zone_importance = full_attn_matrix[zs:ze].mean(axis=0)
        zone_cats = _categorize_tokens(tokens, zone_importance, categories_order=key_categories_all)
        zone_cat_data[zname] = {k: np.mean(v) if v else 0 for k, v in zone_cats.items()}

    key_categories = [cat for cat in key_categories_all
                      if any(zone_cat_data[z].get(cat, 0) > 0 for z in zone_names)]

    x_zones = np.arange(len(zone_names))
    n_cats_d = len(key_categories)
    bar_width = 0.8 / max(n_cats_d, 1)
    cat_colors = {
        'Branches': '#762a83', 'Aromatic': '#fee08b', 'Bonds': '#fc8d59',
        'Metal': '#d73027', 'MOFid': '#4575b4', 'Atoms': '#91bfdb',
    }

    for i, cat in enumerate(key_categories):
        vals = [zone_cat_data[z].get(cat, 0) for z in zone_names]
        offset = (i - (n_cats_d - 1) / 2) * bar_width
        ax_d.bar(x_zones + offset, vals, bar_width,
                 label=cat, color=cat_colors.get(cat, '#999999'),
                 edgecolor='black', linewidth=0.3, alpha=0.85)

    ax_d.set_xticks(x_zones)
    ax_d.set_xticklabels(zone_names, fontsize=SMALL_FS)
    ax_d.set_ylabel('Avg Attn', fontsize=SMALL_FS)
    ax_d.set_title('d', fontsize=BODY_FS, loc='left', pad=4)
    ymax = max(max(zone_cat_data[z].get(cat, 0) for cat in key_categories)
               for z in zone_names)
    ax_d.set_ylim(0, ymax * 1.45)
    # Legend is multi-column and sits in a narrow panel; SMALL_FS is the
    # largest size that does NOT overlap the bars.
    ax_d.legend(fontsize=SMALL_FS, ncol=min(n_cats_d, 3), framealpha=0.8,
                edgecolor='grey', columnspacing=0.6,
                handlelength=1.2, handletextpad=0.3,
                loc='upper right')
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)
    ax_d.tick_params(axis='both', labelsize=SMALL_FS)

    # =========================================================================
    # 9. Final adjustments
    #    Larger left/bottom margins accommodate Transformer-Layer label and
    #    rotated x-tick token labels without overlap.
    # =========================================================================
    plt.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.12)

    # Post-GridSpec fine-tuning (user-tuned visual balance):
    #   * Shift panels (b, c, d) slightly to the RIGHT so the right column
    #     is visually separated from the heatmap instead of crowding it.
    #   * Panel (d) keeps its vertical position; panels (b) and (c) each
    #     slide DOWN so the (b,c,d) stack becomes more compact without
    #     changing any axes size.
    # All values are figure-fractional coordinates.
    dx_right = 0.022  # horizontal shift applied to b, c, and d alike
    dy_b = 0.032      # b moves down the most (far from d)
    dy_c = 0.016      # c moves down half as much (middle of the stack)
    for ax_rc, dy in ((ax_b, dy_b), (ax_c, dy_c), (ax_d, 0.0)):
        pos = ax_rc.get_position()
        ax_rc.set_position((pos.x0 + dx_right, pos.y0 - dy,
                            pos.width, pos.height))

    _save_figure(fig, output_path)

    def _attention_entropy(attn_tensor):
        attn_np = attn_tensor.float().cpu().numpy()
        attn_np = attn_np / attn_np.sum(axis=-1, keepdims=True)
        attn_np = np.clip(attn_np, 1e-10, 1)
        entropy = -np.sum(attn_np * np.log(attn_np), axis=-1)
        return entropy.mean()

    entropy_stats = {
        'canonical_entropy': float(_attention_entropy(c_avg)),
        'isomeric_entropy': float(_attention_entropy(i_avg)),
    }

    return dtw_results, feature_results, stats_b, entropy_stats

# ============================================================================
# Figure 8: Combined 4-Panel Figure for Paper
# ============================================================================
# ...existing code...
def plot_combined_attention_figure(
    attentions: Tuple[torch.Tensor],
    deep_attentions: Tuple[torch.Tensor],
    tokens: List[str],
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    canonical_input: str,
    isomeric_input: str,
    output_path: str,
    deep_start: int,
    num_layers: int,
    device: str = "cpu",
):
    """
    Combined 4-panel figure (2x2) for the attention analysis section.
    
    (a) Token-level attention importance (deep layers) — column-wise mean
    (b) SMILES feature category attention (deep layers)
    (c) Canonical vs Isomeric comparison
    (d) Cross-layer token importance — per-layer row-normalized
    
    Returns DTW analysis results for inclusion in summary JSON.
    """
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.75))
    ((ax_a, ax_b), (ax_c, ax_d)) = axes
    
    display_tokens = [_clean_token_label(t) for t in tokens]
    n = len(display_tokens)
    
    # ---- (a) Token Importance (deep layers) ----
    # Use column-wise MEAN (consistent with panel d) instead of sum
    all_attn_deep = torch.stack([attn[0] for attn in deep_attentions])
    avg_attn_matrix = all_attn_deep.mean(dim=(0, 1)).float().cpu().numpy()  # (seq, seq)
    token_importance = avg_attn_matrix.mean(axis=0)  # column-wise mean -> (seq,)
    token_importance_norm = token_importance / token_importance.max()
    
    colors_a = plt.cm.YlOrRd(token_importance_norm)
    x = np.arange(n)
    ax_a.bar(x, token_importance, color=colors_a, edgecolor='grey',
             linewidth=0.3, width=0.8)
    ax_a.set_xticks(x)
    # 2×2 grid halves panel width; many rotated tokens need TINY_FS to fit
    a_tick_fs = TINY_FS if n > 25 else SMALL_FS
    ax_a.set_xticklabels(display_tokens, rotation=90, ha='center', fontsize=a_tick_fs)
    ax_a.set_ylabel('Avg Attn Received', fontsize=SMALL_FS)
    ax_a.set_title('a', fontsize=PANEL_FS, loc='left', pad=4)
    ax_a.set_xlim(-0.5, n - 0.5)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.tick_params(axis='y', labelsize=SMALL_FS)
    
    # ---- (b) SMILES Feature Attention (deep layers) ----
    categories = _categorize_tokens(tokens, token_importance)
    
    stats = {}
    for cat, values in categories.items():
        if values:
            stats[cat] = {'mean': np.mean(values), 'std': np.std(values), 'count': len(values)}
    
    cat_names = list(stats.keys())
    cat_means = [stats[c]['mean'] for c in cat_names]
    cat_stds = [stats[c]['std'] for c in cat_names]
    n_cats = len(cat_names)
    xb = np.arange(n_cats)
    colors_b = ['#d73027', '#fc8d59', '#fee08b', '#91bfdb', '#4575b4',
                '#762a83', '#1b7837', '#b15928']
    bars = ax_b.bar(xb, cat_means, yerr=cat_stds, capsize=2,
                    color=colors_b[:n_cats], alpha=0.85,
                    edgecolor='black', linewidth=0.4, width=0.6,
                    error_kw={'linewidth': 0.5, 'capthick': 0.5})
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(cat_names, fontsize=SMALL_FS, rotation=20, ha='right')
    ax_b.set_ylabel('Avg Attention', fontsize=SMALL_FS)
    ax_b.set_title('b', fontsize=BODY_FS, loc='left', pad=4)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.tick_params(axis='y', labelsize=SMALL_FS)
    for i, bar in enumerate(bars):
        bar_height = bar.get_height()
        label_y = bar_height * 0.4
        ax_b.text(bar.get_x() + bar.get_width() / 2., label_y,
                  f'n={int(stats[cat_names[i]]["count"])}',
                  ha='center', va='center', fontsize=TINY_FS, color='white',
                  bbox=dict(boxstyle='round,pad=0.15', facecolor='black',
                            alpha=0.5, linewidth=0))
      
    # ---- (c) Canonical vs Isomeric ----
    canonical_attn, canonical_tokens_raw, _, _ = get_attention_weights(
        model, tokenizer, canonical_input, device)
    isomeric_attn, isomeric_tokens_raw, _, _ = get_attention_weights(
        model, tokenizer, isomeric_input, device)
    
    c_avg = torch.stack([attn[0] for attn in canonical_attn]).mean(dim=(0, 1))
    i_avg = torch.stack([attn[0] for attn in isomeric_attn]).mean(dim=(0, 1))
    # Use column-wise mean (consistent with panel a/d)
    c_importance = c_avg.float().cpu().numpy().mean(axis=0)
    i_importance = i_avg.float().cpu().numpy().mean(axis=0)
    c_norm = c_importance / c_importance.max()
    i_norm = i_importance / i_importance.max()
    
    c_tokens_clean = [_clean_token_label(t) for t in canonical_tokens_raw]
    i_tokens_clean = [_clean_token_label(t) for t in isomeric_tokens_raw]
    
    ax_c.plot(np.arange(len(c_norm)), c_norm, color='#d73027', linewidth=0.8,
              marker='o', markersize=2, label=f'Canonical ({len(c_norm)} tokens)',
              alpha=0.85)
    ax_c.plot(np.arange(len(i_norm)), i_norm, color='#4575b4', linewidth=0.8,
              marker='s', markersize=2, label=f'Isomeric ({len(i_norm)} tokens)',
              alpha=0.85)
    ax_c.set_xlabel("Token Position (each sequence's own index)", fontsize=SMALL_FS)
    ax_c.set_ylabel('Normalized Attention', fontsize=SMALL_FS)
    ax_c.set_title('c', fontsize=BODY_FS, loc='left', pad=4)
    ax_c.legend(fontsize=SMALL_FS, loc='upper right', framealpha=0.8)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.tick_params(axis='both', labelsize=SMALL_FS)
    ax_c.set_ylim(-0.05, 1.20)

    # Compute DTW analysis
    dtw_results = compute_dtw_analysis(c_importance, i_importance,
                                        canonical_tokens_raw, isomeric_tokens_raw)
    feature_results = compute_feature_level_correlation(c_importance, i_importance,
                                                         canonical_tokens_raw, isomeric_tokens_raw)
    
    # Narrow sub-panel in 2×2 grid — TINY_FS needed to keep this 2-line
    # statistics box from spilling across the plotted lines
    dtw_text = (f"DTW $r$ = {dtw_results['pearson_r_aligned']:.2f} "
                f"($p$ = {dtw_results['pearson_p_aligned']:.1e})\n"
                f"Random baseline $r$ = {dtw_results['random_baseline_pearson_mean']:.2f} "
                f"\u00b1 {dtw_results['random_baseline_pearson_std']:.2f}")
    ax_c.text(0.03, 0.05, dtw_text, transform=ax_c.transAxes, fontsize=TINY_FS,
              verticalalignment='bottom',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
    
    # ---- (d) Cross-Layer Attention Dynamics (row-normalized) ----
    total_layers = len(attentions)
    if total_layers <= 8:
        select_layers = list(range(total_layers))
    else:
        step = total_layers // 6
        select_layers = sorted(set([0, step, 2*step, total_layers//2,
                                     total_layers - step, total_layers - 2, total_layers - 1]))
    
    attention_matrix = []
    layer_labels = []
    for li in select_layers:
        layer_attn = attentions[li][0].mean(dim=0).float().cpu().numpy()  # (seq, seq)
        token_received = layer_attn.mean(axis=0)  # (seq,)
        attention_matrix.append(token_received)
        layer_labels.append(f'L{li}')
    attention_matrix = np.array(attention_matrix)

    # === Row-wise normalization: each layer independently scaled to [0, 1] ===
    row_min = attention_matrix.min(axis=1, keepdims=True)
    row_max = attention_matrix.max(axis=1, keepdims=True)
    row_range = row_max - row_min
    row_range[row_range == 0] = 1.0  # avoid division by zero
    attention_matrix_norm = (attention_matrix - row_min) / row_range

    im = ax_d.imshow(attention_matrix_norm, cmap='YlOrRd',
                     aspect='auto', interpolation='nearest')
    ax_d.set_xticks(np.arange(n))
    d_tick_fs = TINY_FS if n > 25 else SMALL_FS
    ax_d.set_xticklabels(display_tokens, rotation=90, ha='center', fontsize=d_tick_fs)
    ax_d.set_yticks(np.arange(len(layer_labels)))
    ax_d.set_yticklabels(layer_labels, fontsize=SMALL_FS)
    ax_d.set_xlabel('Token', fontsize=SMALL_FS)
    ax_d.set_ylabel('Layer', fontsize=SMALL_FS)
    ax_d.set_title('d', fontsize=BODY_FS, loc='left', pad=4)

    cbar = fig.colorbar(im, ax=ax_d, shrink=0.8, pad=0.03, aspect=25)
    cbar.set_label('Norm. Attn (per-layer)', fontsize=SMALL_FS)
    cbar.ax.tick_params(labelsize=SMALL_FS, width=0.4)
    cbar.outline.set_linewidth(0.4)

    plt.tight_layout(h_pad=2.0, w_pad=1.5)
    _save_figure(fig, output_path)
    
    return dtw_results, feature_results, stats

def main():
    parser = argparse.ArgumentParser(description='Publication-Quality Attention Visualization')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--adapter_path', type=str, default=None)
    parser.add_argument('--smiles', type=str, default='CC(=O)OC1=CC=CC=C1C(=O)O')
    parser.add_argument('--isomeric_smiles', type=str, default=None)
    parser.add_argument('--mofid', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='./attention_plots_paper')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--layer', type=int, default=None,
                        help='Specific layer to visualize heads')
    parser.add_argument('--sample_name', type=str, default='sample')
    parser.add_argument('--deep_layers_only', action='store_true',
                        help='Only use the last 25%% layers for averaging (recommended for papers)')
    parser.add_argument('--deep_layer_start', type=int, default=None,
                        help='Start layer index for deep-layer averaging (e.g., 24 for 32-layer model)')
    parser.add_argument('--control_smiles', type=str, default=None,
                        help='Control SMILES for cross-molecule invariance validation. '
                             'Multiple entries separated by ";;". Each entry is a full '
                             'input string (SMILES + MOFid if applicable).')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.adapter_path, args.device)

    input_text = f"{args.smiles} {args.mofid}" if args.mofid else args.smiles
    print(f"\nAnalyzing: {input_text[:100]}...")

    attentions, tokens, hidden_states, valid_indices = get_attention_weights(
        model, tokenizer, input_text, args.device
    )

    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    print(f"Layers: {num_layers}, Heads: {num_heads}, Tokens: {len(tokens)}")
    print(f"Tokens: {tokens}")

    # Determine deep layer range
    if args.deep_layer_start is not None:
        deep_start = args.deep_layer_start
    else:
        deep_start = max(0, num_layers - num_layers // 4)  # Last 25% layers

    # Deep layers only: slice attention tuple
    deep_attentions = attentions[deep_start:]
    print(f"Deep layers: {deep_start}-{num_layers-1} ({len(deep_attentions)} layers)")

    print("\n" + "=" * 50)
    print("Generating publication-quality figures...")
    print("=" * 50)

    # ---- Standalone figures (Skip if doing comparison to avoid redundancy) ----
    if not args.isomeric_smiles:
        plot_avg_attention_heatmap(attentions, tokens, os.path.join(args.output_dir, f'{args.sample_name}_avg_attention.png'), title='Average Attention Across All Layers and Heads')
        plot_token_importance(attentions, tokens, os.path.join(args.output_dir, f'{args.sample_name}_token_importance.png'), title='Token Importance (Attention Received)')
        plot_avg_attention_heatmap(deep_attentions, tokens, os.path.join(args.output_dir, f'{args.sample_name}_avg_attention_deep.png'), title=f'Average Attention (Layer {deep_start}\u2013{num_layers-1})')
        plot_token_importance(deep_attentions, tokens, os.path.join(args.output_dir, f'{args.sample_name}_token_importance_deep.png'), title=f'Token Importance (Layer {deep_start}\u2013{num_layers-1})')
        plot_first_token_attention(attentions, tokens, os.path.join(args.output_dir, f'{args.sample_name}_first_token_attention.png'), title='Cross-Layer Token Importance')
        
        stats = plot_smiles_feature_attention(attentions, tokens, os.path.join(args.output_dir, f'{args.sample_name}_smiles_feature_attention.png'), title='Attention by SMILES Feature Type')
        stats_deep = plot_smiles_feature_attention(deep_attentions, tokens, os.path.join(args.output_dir, f'{args.sample_name}_smiles_feature_attention_deep.png'), title=f'Attention by SMILES Feature (Layer {deep_start}\u2013{num_layers-1})')
    else:
        print("Skipping redundant standalone plots (They are included in the Flagship Combined Figures).")
        # Manually compute stats for JSON summary to avoid drawing the redundant plot
        all_attn_matrix = torch.stack([a[0] for a in attentions]).mean(dim=(0, 1)).float().cpu().numpy().mean(axis=0)
        categories_all = _categorize_tokens(tokens, all_attn_matrix)
        stats = {cat: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'count': len(v)} for cat, v in categories_all.items() if v}

        deep_attn_matrix = torch.stack([a[0] for a in deep_attentions]).mean(dim=(0, 1)).float().cpu().numpy().mean(axis=0)
        categories_deep = _categorize_tokens(tokens, deep_attn_matrix)
        stats_deep = {cat: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'count': len(v)} for cat, v in categories_deep.items() if v}

    # Fig 4: Selected layer patterns (Not redundant, keep it)
    plot_selected_layer_patterns(
        attentions, tokens,
        os.path.join(args.output_dir, f'{args.sample_name}_layer_patterns.png'),
        title='Attention Patterns Across Selected Layers'
    )

    # Fig 5: Selected head patterns (Not redundant, keep it)
    target_layer = args.layer if args.layer is not None else num_layers - 1
    plot_selected_head_patterns(
        attentions, tokens, target_layer,
        os.path.join(args.output_dir, f'{args.sample_name}_layer{target_layer}_heads.png'),
        title=f'Attention Heads in Layer {target_layer}'
    )

    # Fig 7-9: Canonical vs Isomeric comparison & FLAGSHIP figures
    if args.isomeric_smiles:
        canonical_input = input_text
        isomeric_input = f"{args.isomeric_smiles} {args.mofid}" if args.mofid else args.isomeric_smiles

        # Entropy calculation handled dynamically now. We skip `plot_canonical_vs_isomeric` standalone image!
        
        # ---- Combined 4-panel figure ----
        print("\n" + "=" * 50)
        print("Generating combined 4-panel figure...")
        print("=" * 50)
        dtw_results, feature_results, panel_stats = plot_combined_attention_figure(
            attentions=attentions,
            deep_attentions=deep_attentions,
            tokens=tokens,
            model=model,
            tokenizer=tokenizer,
            canonical_input=canonical_input,
            isomeric_input=isomeric_input,
            output_path=os.path.join(args.output_dir, f'{args.sample_name}_combined_attention.png'),
            deep_start=deep_start,
            num_layers=num_layers,
            device=args.device,
        )

        print("\n--- DTW Analysis Results ---")
        print(f"  DTW distance (raw):        {dtw_results['dtw_distance']:.4f}")
        print(f"  DTW distance (normalized): {dtw_results['dtw_distance_normalized']:.4f}")
        print(f"  Alignment path length:     {dtw_results['path_length']}")
        print(f"  Pearson r (aligned):       {dtw_results['pearson_r_aligned']:.4f} "
              f"(p = {dtw_results['pearson_p_aligned']:.2e})")
        print(f"  Spearman r (aligned):      {dtw_results['spearman_r_aligned']:.4f} "
              f"(p = {dtw_results['spearman_p_aligned']:.2e})")
        print(f"  Random baseline r:         {dtw_results['random_baseline_pearson_mean']:.4f} "
              f"± {dtw_results['random_baseline_pearson_std']:.4f}")
        print(f"  Permutation p-value:       {dtw_results['p_value_permutation']:.4f} "
              f"(n = {dtw_results['n_permutations']})")

        print("\n--- Feature-Level Correlation ---")
        print(f"  Pearson r (features):      {feature_results['feature_pearson_r']:.4f} "
              f"(p = {feature_results['feature_pearson_p']:.4f})")
        print(f"  Spearman r (features):     {feature_results['feature_spearman_r']:.4f} "
              f"(p = {feature_results['feature_spearman_p']:.4f})")
        for cat in feature_results['categories']:
            c_val = feature_results['canonical_feature_attention'][cat]
            i_val = feature_results['isomeric_feature_attention'][cat]
            print(f"    {cat:20s}: canonical={c_val:.4f}, isomeric={i_val:.4f}")

        # ---- Hierarchical Attention Overview (new flagship figure) ----
        print("\n" + "=" * 50)
        print("Generating hierarchical attention overview...")
        print("=" * 50)
        hier_dtw, hier_feat, hier_stats, hier_entropy = plot_hierarchical_attention_overview(
            attentions=attentions,
            tokens=tokens,
            model=model,
            tokenizer=tokenizer,
            canonical_input=canonical_input,
            isomeric_input=isomeric_input,
            output_path=os.path.join(args.output_dir, f'{args.sample_name}_hierarchical_overview.png'),
            device=args.device,
            deep_start=deep_start,
            num_layers=num_layers,
        )
        print("Hierarchical overview figure generated.")
        entropy_stats = hier_entropy
    else:
        dtw_results = None
        feature_results = None
        entropy_stats = None
        hier_dtw = None
        hier_feat = None
        hier_stats = None
        hier_entropy = None

    # =========================================================================
    # Isomeric SMILES independent analysis (token classification & per-layer)
    # =========================================================================
    isomeric_analysis = None
    if args.isomeric_smiles:
        iso_input = f"{args.isomeric_smiles} {args.mofid}" if args.mofid else args.isomeric_smiles
        iso_attn, iso_tokens, _, _ = get_attention_weights(model, tokenizer, iso_input, args.device)
        iso_n_layers = len(iso_attn)
        iso_n_tokens = len(iso_tokens)

        # Token classification
        iso_classes = [_classify_smiles_token(t) for t in iso_tokens]
        iso_class_counts = {}
        for c in iso_classes:
            iso_class_counts[c] = iso_class_counts.get(c, 0) + 1

        # Full-layer attention matrix (same method as canonical)
        iso_full_attn = np.zeros((iso_n_layers, iso_n_tokens))
        for li in range(iso_n_layers):
            la = iso_attn[li][0].mean(dim=0).float().cpu().numpy()
            iso_full_attn[li] = la.mean(axis=0)

        # All-layer feature stats
        iso_importance_all = iso_full_attn.mean(axis=0)
        iso_cats_all = _categorize_tokens(iso_tokens, iso_importance_all)
        iso_stats_all = {}
        for cat, vals in iso_cats_all.items():
            if vals:
                iso_stats_all[cat] = {'mean': float(np.mean(vals)),
                                       'std': float(np.std(vals)),
                                       'count': len(vals)}

        # Deep-layer feature stats
        iso_deep_start = iso_n_layers - max(1, iso_n_layers // 4)
        iso_importance_deep = iso_full_attn[iso_deep_start:].mean(axis=0)
        iso_cats_deep = _categorize_tokens(iso_tokens, iso_importance_deep)
        iso_stats_deep = {}
        for cat, vals in iso_cats_deep.items():
            if vals:
                iso_stats_deep[cat] = {'mean': float(np.mean(vals)),
                                        'std': float(np.std(vals)),
                                        'count': len(vals)}

        # Per-layer-zone feature breakdown (same 3 zones)
        iso_zone_ranges = [(0, 10), (10, 20), (20, iso_n_layers)]
        iso_zone_names = ['L0-10_local', 'L11-20_functional', f'L21-{iso_n_layers-1}_topology']
        iso_zone_features = {}
        key_cats = ['Ring Closures', 'Branches', 'Bonds', 'Aromatic', 'Metal', 'MOFid', 'Atoms']
        for zname, (zs, ze) in zip(iso_zone_names, iso_zone_ranges):
            zone_imp = iso_full_attn[zs:ze].mean(axis=0)
            zone_cats = _categorize_tokens(iso_tokens, zone_imp, categories_order=key_cats)
            iso_zone_features[zname] = {k: float(np.mean(v)) if v else 0.0
                                         for k, v in zone_cats.items()}

        isomeric_analysis = {
            'isomeric_input_text': iso_input,
            'isomeric_seq_length': iso_n_tokens,
            'isomeric_tokens': iso_tokens,
            'isomeric_token_classification': iso_class_counts,
            'isomeric_feature_stats_all_layers': iso_stats_all,
            'isomeric_feature_stats_deep_layers': iso_stats_deep,
            'isomeric_deep_layer_range': f'{iso_deep_start}-{iso_n_layers-1}',
            'isomeric_per_zone_feature_attention': iso_zone_features,
        }

        print("\n--- Isomeric SMILES Analysis ---")
        print(f"  Input:  {iso_input}")
        print(f"  Tokens: {iso_n_tokens}")
        print(f"  Token classification: {iso_class_counts}")
        print(f"  Deep-layer range: {iso_deep_start}-{iso_n_layers-1}")
        for cat, st in iso_stats_deep.items():
            print(f"    {cat:20s}: mean={st['mean']:.4f}, std={st['std']:.4f}, n={st['count']}")

    # =========================================================================
    # Cross-molecule control analysis (representation invariance validation)
    # =========================================================================
    cross_mol_results = None
    if args.control_smiles and args.isomeric_smiles:
        control_inputs = [s.strip() for s in args.control_smiles.split(';;')
                          if s.strip()]
        if control_inputs:
            print("\n" + "=" * 60)
            print("Cross-molecule control analysis for invariance validation")
            print("=" * 60)
            canonical_input = input_text

            cross_mol_results = compute_cross_molecule_control(
                model, tokenizer, canonical_input, control_inputs, args.device)

            # Comparison figure
            plot_invariance_control_comparison(
                same_mol_dtw_r=dtw_results['pearson_r_aligned'],
                same_mol_feat_r=feature_results['feature_pearson_r'],
                cross_mol_results=cross_mol_results,
                random_baseline_r=dtw_results['random_baseline_pearson_mean'],
                random_baseline_std=dtw_results['random_baseline_pearson_std'],
                output_path=os.path.join(
                    args.output_dir,
                    f'{args.sample_name}_invariance_control.png'),
            )

            # Formatted output
            print("\n" + "-" * 70)
            print("  REPRESENTATION INVARIANCE: Same-Molecule vs Cross-Molecule")
            print("-" * 70)
            print(f"  {'Comparison':<35s} {'DTW r':>8s} {'Feat r':>8s}")
            print(f"  {'─' * 35} {'─' * 8} {'─' * 8}")
            print(f"  {'Same mol. (Can. vs Iso.)':<35s} "
                  f"{dtw_results['pearson_r_aligned']:>8.4f} "
                  f"{feature_results['feature_pearson_r']:>8.4f}")
            for i, comp in enumerate(cross_mol_results['comparisons']):
                label = f"Control {i+1}: {comp['control_input'][:25]}..."
                print(f"  {label:<35s} "
                      f"{comp['dtw_pearson_r']:>8.4f} "
                      f"{comp['feature_pearson_r']:>8.4f}")
            print(f"  {'Random baseline':<35s} "
                  f"{dtw_results['random_baseline_pearson_mean']:>8.4f} "
                  f"{'N/A':>8s}")
            print(f"  {'─' * 35} {'─' * 8} {'─' * 8}")
            print(f"  {'Cross-mol. mean ± std (DTW)':<35s} "
                  f"{cross_mol_results['cross_molecule_dtw_r_mean']:.4f}"
                  f"±{cross_mol_results['cross_molecule_dtw_r_std']:.4f}")
            print(f"  {'Cross-mol. mean ± std (Feat)':<35s} "
                  f"{'':>8s} "
                  f"{cross_mol_results['cross_molecule_feat_r_mean']:.4f}"
                  f"±{cross_mol_results['cross_molecule_feat_r_std']:.4f}")

            same_dtw = dtw_results['pearson_r_aligned']
            cross_dtw = cross_mol_results['cross_molecule_dtw_r_mean']
            delta = same_dtw - cross_dtw
            print(f"\n  *** Δ(same - cross) DTW r = {delta:.4f} ***")
            if delta > 0.1:
                print("  → Same-molecule correlation substantially exceeds "
                      "cross-molecule baseline.")
                print("    Representation invariance is supported beyond "
                      "universal MOF SMILES properties.")
            elif delta > 0.05:
                print("  → Moderate difference; invariance partially supported.")
            else:
                print("  → Small difference; category-level similarity may "
                      "largely reflect universal MOF SMILES properties.")
            print("-" * 70)

    # =========================================================================
    # Canonical token classification (for JSON symmetry)
    # =========================================================================
    can_classes = [_classify_smiles_token(t) for t in tokens]
    can_class_counts = {}
    for c in can_classes:
        can_class_counts[c] = can_class_counts.get(c, 0) + 1

    # Canonical per-layer-zone feature breakdown
    can_full_attn = np.zeros((num_layers, len(tokens)))
    for li in range(num_layers):
        la = attentions[li][0].mean(dim=0).float().cpu().numpy()
        can_full_attn[li] = la.mean(axis=0)

    can_zone_ranges = [(0, 10), (10, 20), (20, num_layers)]
    can_zone_names = ['L0-10_local', 'L11-20_functional', f'L21-{num_layers-1}_topology']
    can_zone_features = {}
    key_cats = ['Ring Closures', 'Branches', 'Bonds', 'Aromatic', 'Metal', 'MOFid', 'Atoms']
    for zname, (zs, ze) in zip(can_zone_names, can_zone_ranges):
        zone_imp = can_full_attn[zs:ze].mean(axis=0)
        zone_cats = _categorize_tokens(tokens, zone_imp, categories_order=key_cats)
        can_zone_features[zname] = {k: float(np.mean(v)) if v else 0.0
                                     for k, v in zone_cats.items()}

    # Save summary
    summary = {
        'input_text': input_text,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'seq_length': len(tokens),
        'tokens': tokens,
        'token_classification': can_class_counts,
        'deep_layer_range': f'{deep_start}-{num_layers-1}',
        'smiles_feature_stats_all_layers': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in stats.items()},
        'smiles_feature_stats_deep_layers': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in stats_deep.items()},
        'canonical_per_zone_feature_attention': can_zone_features,
        'dtw_analysis': dtw_results,
        'feature_level_correlation': feature_results,
        'entropy_comparison': entropy_stats,
        'isomeric_analysis': isomeric_analysis,
        'cross_molecule_control': cross_mol_results,
        'hierarchical_overview': {
            'dtw_analysis': hier_dtw,
            'feature_level_correlation': hier_feat,
            'feature_stats_deep_layers': {
                k: {kk: float(vv) for kk, vv in v.items()}
                for k, v in (hier_stats or {}).items()
            } if hier_stats else None,
            'entropy_comparison': hier_entropy,
        } if hier_dtw is not None else None,
        'figure_settings': {
            'dpi': DPI,
            'single_col_width_in': SINGLE_COL_WIDTH,
            'double_col_width_in': DOUBLE_COL_WIDTH,
            'max_height_in': MAX_HEIGHT,
            'colormap': 'YlOrRd',
            'font_family': 'Arial (sans-serif, non-bold, body size 7 pt)',
            'save_formats': SAVE_FORMATS,
            'nature_compliant': True,
        }
    }
    summary_path = os.path.join(args.output_dir, f'{args.sample_name}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {summary_path}")
    print(f"All figures saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
