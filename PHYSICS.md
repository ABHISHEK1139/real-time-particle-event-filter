# Standard Model Physics Context (Z Boson $\rightarrow \mu^+ \mu^-$)

This project is a **supervised demonstration of learning a Z-mass-window
selection from muon kinematics** on educational open data — not a
trigger/discovery system.

## Dataset (fixed identity)

- **CMS DoubleMu Run2011A, proton-proton collisions at 7 TeV.**
- Educational 100k-event derived sample: <https://opendata.cern.ch/record/5201>
  (parent: *Datasets derived from the Run2011A … DoubleMu primary datasets*,
  <https://opendata.cern.ch/record/545>).
- Previously the repo called this "Run2010B" and cited 13.6 TeV. Both were
  wrong for this file: the Run2010B dimuon sample is a different record
  (<https://opendata.cern.ch/record/700>), and 13.6 TeV is Run-3 energy,
  whereas 2010/2011 Run-1 data are 7 TeV pp collisions.
- CERN flags the 5201 sample as education/outreach, **not suitable for a full
  physics analysis**. Treat all numbers here as ML-pipeline figures on that
  curated subset, not detector performance.

## Physical Relevance

At the LHC Run-1 energy of **7 TeV** (this data), most collisions produce
low-momentum QCD background; a small fraction produce electroweak bosons such
as the $Z$ ($\approx 91.18$ GeV) and $W^\pm$ ($\approx 80.4$ GeV).

The $Z$ boson (lifetime $\sim 3 \times 10^{-25}$ s) is seen only via decay
products — here $Z \rightarrow \mu^+ \mu^-$ ($\approx 3.3\%$ of decays).

## The Drell-Yan Background

The main continuum under the $Z$ peak is the **Drell-Yan process**
($q\bar q \to \gamma^*/Z \to \ell^+\ell^-$), plus low-mass resonances
($J/\psi$, $\Upsilon$). The educational sample is already a clean,
preselected dimuon subset, so separation here is far easier than in a raw
trigger environment.

## Mathematical Kinematics

For a dimuon pair:

$$ M = \sqrt{2 p_{T1} p_{T2} (\cosh(\eta_1 - \eta_2) - \cos(\phi_1 - \phi_2))} $$

- $p_{T}$: transverse momentum; $\eta$: pseudorapidity; $\phi$: azimuthal angle.

Labels in this repo are `1 if 80 < M < 100 else 0`, while model inputs are
($p_T, \eta, \phi$) — which **mathematically determine M** (the GNN additionally
sees $E, p_x, p_y, p_z$, from which $M$ is directly reconstructible). By
excluding $M$ as a feature we force the model to *re-learn that geometric
relation*, which is pedagogically interesting but is **not** evidence of
trigger-level discovery power. The honest claim is mass-window learning, and
accuracy must be read alongside AUROC/AUPRC given the ~95/5 class imbalance.
