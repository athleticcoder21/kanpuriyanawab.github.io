# AR RL → diffusion worklog

Edit `index.qmd`; all charts read the local exports in `data/`. The original
import in `blog/` is retained as a reference and excluded from site rendering.
No GPU or W&B credentials are needed.

From the repository root, set up the plotting environment and render:

```sh
bash posts/rl_ar_diffusion/setup.sh
bash posts/rl_ar_diffusion/setup.sh --preview
```

The script sets `QUARTO_PYTHON` for both render and live preview. A plain
`quarto preview` can select a different Python environment without Plotly.

Open `/posts/rl_ar_diffusion/` in the preview. Commit the updated post, data,
and `_freeze/posts/rl_ar_diffusion/` after rendering. The frozen HTML includes
Plotly itself and lets the publishing workflow render without Python packages
or network access to a chart CDN. After changing a CSV, explicitly render the
post again to refresh the frozen charts, even if the QMD has not changed.

`modes.csv` supplies checkpoint accuracy, standard errors, truncation counts,
and recorded paired-test p-values. `train.csv` supplies training curves;
`train_eval.csv` supplies the separate sampled trainer check. `probes.csv`
records task-selection and engine probes. The copied export does not include
per-problem outputs or the original training/evaluation code, so claims about
those details remain the experiment author's account.
