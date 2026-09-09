# Publishing this repository

Steps to put the repo on GitHub, archive it on Zenodo with a citable DOI,
and reference it in the paper. Do this once the manuscript is close to
acceptance.

## 0. Before `git init`

This folder currently lives inside OneDrive. Either **move it out of the
OneDrive-synced tree** first, or exclude it from sync (OneDrive settings ->
"Choose folders"). Git and a live file-sync client on the same `.git`
directory can corrupt each other.

Fill in the placeholders:

* `<your-github-username>` - in `README.md`, `pyproject.toml`, `CITATION.cff`,
  `.github/workflows/ci.yml`
* ORCIDs in `CITATION.cff` (uncomment the `orcid:` lines)
* the article DOI in `README.md`, `CITATION.cff`, `.zenodo.json`,
  `pyproject.toml` (once you have it)

## 1. GitHub

```bash
cd sdd-voltammetry
git init
git add -A
git commit -m "Initial public release: SDD/Dunn code, examples, experimental data"

# create the repo and push (needs the GitHub CLI `gh`, or do it in the web UI)
gh repo create sdd-voltammetry --public --source=. --push --description \
  "Semi-Derivative Decomposition of faradaic and capacitive currents in voltammetry"
```

Optional: on GitHub, Settings -> Branches -> protect `main`.
The included `.github/workflows/ci.yml` runs the smoke test on every push.

## 2. Zenodo -> DOI

1. Sign in to https://zenodo.org with your GitHub account.
2. Zenodo -> top-right menu -> **GitHub**; flip the switch next to
   `sdd-voltammetry` to **On**.
3. On GitHub: **Releases -> Draft a new release**, tag `v1.0.0`, title
   *"Code and data for Pianta et al., Separation of Faradaic and Capacitive
   Currents ..."*, publish.
4. Zenodo automatically archives that release and mints a DOI. It also
   creates a **concept DOI** that always points to the latest version -
   **cite the concept DOI in the paper.**
5. Add the badge to `README.md` (uncomment the DOI line, fill the number).
6. If the code changes during proofs, make a `v1.0.1` release - Zenodo
   archives it and updates the concept DOI target automatically.

## 3. Text for the manuscript

**Data availability statement** (replace "available upon reasonable request"
/ "in the SI"):

> The `cv_deconvolution` Python package implementing the SDD and Dunn
> analyses, runnable example scripts that reproduce every figure of this
> work on simulated data, and the experimental voltammetry dataset are
> available at https://github.com/<your-github-username>/sdd-voltammetry and
> archived at Zenodo (DOI: 10.5281/zenodo.XXXXXXX).

**Abstract** - change

> "a Python code has been made publicly available in the SI of this work"

to

> "the code and runnable examples are available in an open-source repository
> (see Data availability)".

**Reply to Reviewer 2** (the code-deposit point, currently unfinished):

> We appreciate this suggestion. We have moved the code out of the PDF: the
> `cv_deconvolution` package, a set of runnable example scripts that
> reproduce each figure of the paper on simulated data, and the experimental
> dataset are now in a public GitHub repository
> (https://github.com/<your-github-username>/sdd-voltammetry), archived on
> Zenodo with a citable DOI (10.5281/zenodo.XXXXXXX). Both are cited in the
> revised Data availability statement.

This also resolves the internal comments about GitHub/Zenodo in the
manuscript margins.
