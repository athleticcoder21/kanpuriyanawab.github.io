"""Generate the worklog's editable SVG diagrams. Uses only Python's standard library."""
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parent
BLUE, ORANGE, INK, MUTED = '#2a78d6', '#eb6834', '#282828', '#626975'


def text(x, y, label, size=16, color=INK, anchor='middle', weight=400):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{color}" font-weight="{weight}">{escape(label)}</text>'


def box(x, y, w, h, fill='#f7f8fa', stroke='#dce0e5', radius=12):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>'


def arrow(path, color=MUTED, dash=False):
    style = ' stroke-dasharray="5 5"' if dash else ''
    return f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" marker-end="url(#arrow)"{style}/>'


def save(name, height, title, description, content):
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 {height}" role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="context-stroke"/></marker></defs>
<rect width="760" height="{height}" fill="white"/>
<g font-family="Arial, Helvetica, sans-serif">{''.join(content)}</g></svg>'''
    (OUT / name).write_text(svg + '\n')


def backbone():
    c = [text(380, 28, 'One model, two decoding modes', 21, weight=600)]
    for x, color, fill, title, subtitle in [
        (30, BLUE, '#edf4fd', 'AR mode', 'Causal attention'),
        (410, ORANGE, '#fff2eb', 'Diffusion mode', 'Attention within a block'),
    ]:
        c += [box(x, 55, 320, 68, fill, color), text(x + 160, 83, title, 18, color, weight=600),
              text(x + 160, 107, subtitle, 15, MUTED)]
    c += [arrow('M190 123 V145 H350 V173', BLUE), arrow('M570 123 V145 H410 V173', ORANGE)]
    c += [box(180, 174, 400, 253), text(380, 201, 'SHARED WEIGHTS', 13, MUTED, weight=600)]
    for y, title, subtitle in [(215, 'Token embeddings', 'Token IDs → vectors'),
                                (281, 'Transformer blocks', 'Attention + feed-forward layers'),
                                (347, 'Vocabulary projection', 'Hidden states → token scores')]:
        c += [box(210, y, 340, 56, '#ffffff'), text(380, y + 23, title, 17, weight=600),
              text(380, y + 44, subtitle, 14, MUTED)]
        if y < 347: c += [arrow(f'M380 {y + 56} V{y + 65}')]
    c += [arrow('M350 428 V450 H190 V475', BLUE), arrow('M410 428 V450 H570 V475', ORANGE)]
    for x, color, fill, title, subtitle in [
        (30, BLUE, '#edf4fd', 'Pick the next token', 'Repeat left to right'),
        (410, ORANGE, '#fff2eb', 'Fill masked positions', 'Repeat until the block is complete'),
    ]:
        c += [box(x, 476, 320, 67, fill, color), text(x + 160, 504, title, 17, color, weight=600),
              text(x + 160, 528, subtitle, 14, MUTED)]
    save('shared_model.svg', 561, 'Shared transformer, different attention and decoding',
         'AR and diffusion select different attention patterns but use the same token embeddings, transformer blocks and vocabulary projection. AR picks one next token; diffusion fills masked positions over multiple passes.', c)


def masks():
    c = [text(380, 27, 'What can each position attend to?', 21, weight=600)]
    n, cell, top = 8, 31, 114
    for left, mode, color in [(82, 'AR · causal', BLUE), (444, 'Diffusion · block attention', ORANGE)]:
        c += [text(left + 124, 64, mode, 18, color, weight=600),
              text(left + 124, 91, 'Position being attended to →', 13, MUTED)]
        for i in range(n):
            c += [text(left + i * cell + cell/2, top - 7, str(i+1), 11, MUTED),
                  text(left - 12, top + i * cell + 20, str(i+1), 11, MUTED)]
            for j in range(n):
                allowed = j <= i if left == 82 else (j <= i if i < 4 else True)
                fill = color if allowed else '#f2f3f5'
                c += [f'<rect x="{left + j*cell}" y="{top + i*cell}" width="{cell-2}" height="{cell-2}" rx="3" fill="{fill}"/>']
        split = left + 4 * cell - 1
        c += [f'<path d="M{split} {top-2} V{top+8*cell} M{left-2} {top+4*cell-1} H{left+8*cell}" stroke="#8b929e" stroke-dasharray="4 4" fill="none"/>',
              text(left + 62, 385, 'Completed prefix', 12, MUTED), text(left + 186, 385, 'Current block', 12, MUTED)]
    c += [f'<text transform="translate(24 240) rotate(-90)" text-anchor="middle" font-size="13" fill="{MUTED}">Query position</text>',
          box(200, 413, 15, 15, BLUE, BLUE, 3), text(226, 425, 'Allowed', 13, MUTED, 'start'),
          box(385, 413, 15, 15, '#f2f3f5', '#e0e3e7', 3), text(411, 425, 'Blocked', 13, MUTED, 'start')]
    save('attention_masks.svg', 449, 'Causal and block attention masks',
         'Rows are query positions; columns are positions they can attend to. Causal attention allows the diagonal and earlier positions. Diffusion additionally allows all positions inside the current block to attend to each other. Four prefix tokens and a four-position current block are shown.', c)


def workflow():
    c = [text(380, 28, 'Train in AR mode, then test both modes', 21, weight=600)]
    nodes = [
        (18, 'Original model', 'NVIDIA layout', '#f7f8fa', '#dce0e5'),
        (208, 'Export AR layout', 'Same weight values', '#edf4fd', BLUE),
        (398, 'AR-only GRPO', 'Updates the weights', '#edf4fd', BLUE),
        (588, 'Checkpoint', 'Trained weights', '#f7f8fa', '#dce0e5'),
    ]
    for x, title, subtitle, fill, stroke in nodes:
        c += [box(x, 62, 154, 80, fill, stroke), text(x + 77, 94, title, 16, weight=600),
              text(x + 77, 120, subtitle, 13, MUTED)]
    for x in [172, 362, 552]: c += [arrow(f'M{x} 102 H{x+33}')]
    c += [arrow('M665 143 V181 H251 V250', BLUE), arrow('M665 181 H574 V223', ORANGE)]
    c += [box(141, 251, 220, 79, '#edf4fd', BLUE), text(251, 282, 'AR evaluation', 18, BLUE, weight=600),
          text(251, 309, 'vLLM', 15, MUTED)]
    c += [box(464, 224, 220, 65, '#fff2eb', ORANGE), text(574, 251, 'Restore NVIDIA layout', 16, weight=600),
          text(574, 274, 'Same trained weight values', 13, MUTED), arrow('M574 290 V316', ORANGE),
          box(464, 317, 220, 79, '#fff2eb', ORANGE), text(574, 348, 'Diffusion evaluation', 18, ORANGE, weight=600),
          text(574, 375, 'SGLang', 15, MUTED),
          text(251, 366, 'Same test questions', 14, MUTED), text(251, 387, 'at every checkpoint', 14, MUTED)]
    save('training_workflow.svg', 419, 'AR-only training and evaluation in both modes',
         'Export the original model in AR layout without changing weight values. GRPO updates the weights through AR training. Evaluate each trained checkpoint in vLLM for AR, and restore the original NVIDIA layout for diffusion evaluation in SGLang.', c)


if __name__ == '__main__':
    backbone()
    masks()
    workflow()
