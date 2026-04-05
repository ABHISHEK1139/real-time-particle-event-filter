# Standard Model Physics Context (Z Boson $\rightarrow \mu^+ \mu^-$)

This system is engineered to isolate the **Z Boson** elementary particle from chaotic background noise in High Energy Physics collision data, specifically operating on the dimuon decay channel ($Z \rightarrow \mu^+ \mu^-$).

## Physical Relevance

At the Large Hadron Collider (LHC), protons collide at energies reaching $13.6$ TeV. These collisions produce myriad secondary particles, the vast majority of which correspond to low-momentum multi-jet QCD background processes. However, a minute fraction of collisions produce massive electroweak gauge bosons, such as the $Z$ ($\approx 91.18$ GeV) and $W^\pm$ ($\approx 80.4$ GeV) bosons. 

The $Z$ boson is notoriously difficult to detect directly because its mean lifetime is roughly $3 \times 10^{-25}$ seconds. We only observe it indirectly via its decay products. In about $3.3\%$ of cases, it decays cleanly into a muon-antimuon pair.

## The Drell-Yan Background

The primary noise profile interfering with $Z \rightarrow \mu^+ \mu^-$ detection is the **Drell-Yan process**, where a quark of one hadron and an antiquark of another annihilate to create a virtual photon ($\gamma^{*}$) or $Z$ boson, which then decays into a lepton pair. Unlike the sharply peaked resonance of a real $Z$ boson, Drell-Yan creates a continuous exponential background spectrum.

Our Machine Learning architecture is specifically optimized to decouple the distinct topographic signature of the $Z$ boson resonance (peaked at ~91 GeV) from this continuous Drell-Yan noise floor and lower-mass resonances ($J/\psi$, $\Upsilon$), without explicitly being spoon-fed the reconstructed mass calculation.

## Mathematical Kinematics

In the data pipeline, the invariant mass $M$ of a dimuon pair is defined as:
$$ M = \sqrt{2 p_{T1} p_{T2} (\cosh(\eta_1 - \eta_2) - \cos(\phi_1 - \phi_2))} $$

Where:
- $p_{T}$: Transverse Momentum (momentum perpendicular to the beam axis)
- $\eta$: Pseudorapidity (spatial angle relative to the beam axis)
- $\phi$: Azimuthal Angle (radial traversal)

By excluding $M$ from our input feature matrices, our deployed Neural Networks and Decision Trees are forced to learn a statistical boundary natively correlated with the invariant geometric separation of the events natively, simulating the constraints of the **High-Level Trigger (HLT)** systems designed to catch these events synchronously in real-time.
