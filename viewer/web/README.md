---
title: OthelloGPT SAE Feature Viewer
emoji: ⚫
colorFrom: green
colorTo: gray
sdk: static
app_file: index.html
---

# OthelloGPT SAE feature viewer

Place pieces on an Othello board; each move runs OthelloGPT **in your browser**
(no backend) and shows the top SAE feature activations at layer-5 resid_post
for 8 seed-42 SAEs: PAnneal, Standard, GatedSAE, TopK, GatedAnneal, SpaDE,
BatchTopK, CosineSpaDE. Weights are fp16 exports of the research checkpoints;
the JS forward pass is verified against torch (see the research repo's
`viewer/web/test_inference.mjs`).
