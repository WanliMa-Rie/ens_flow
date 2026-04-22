### Ensemble metrics
Ensemble prediction should be evaluated via two orthogonal dimensions: **coverage of GT conformational states** and **quality of generated samples**. Coverage asks whether the GT conformers are covered by at least one generated structure. Quality asks whether generated structures correspond to GT conformers rather than redundant samples (noises).
Therefore, we report both recall-style and precision-style metrics based on RMSD.
- Coverage Recall ( $\text{COV}-R(\delta)$ ): measures the fraction of GT conformers covered by at least on generated samples within an RMSD threshold $\delta$ .
$$
COV-R_{\delta}=\frac{1}{K}\sum_{k=1}^K\mathbb{1}(\min_{x\in Gen}\text{RMSD}(x,GT_k)\leq\delta)
$$
- Average minimum RMSD Recall (AMR-R): measures the average distance from every GT conformers to its nearest generated samples.
$$
AMR-R=\frac{1}{K}\sum_{k=1}^K(\min_{x\in Gen}\text{RMSD}(x,GT_k))
$$
- Coverage Precision (COV-P( $\delta$ )): measures the fraction of generated samples that match at least one GT conformer within threshold $\delta$ .
$$
COV-P_{\delta}=\frac{1}{N}\sum_{i=1}^N\mathbb{1}\left[\min_{k}\text{RMSD}(x_i,GT_k)\leq\delta\right]
$$
- Average minimum RMSD Precision (AMR-P): measures the average distance from each generated conformer to its nearest GT conformer.
$$
AMR-R=\frac{1}{N}\sum_{i=1}^N(\min_{k}\text{RMSD}(x_i,GT_k))
$$

Also, we'll report the diversity of generated samples, to prevent the modal collapse.
- Pairwise RMSD: measures the diversity within the generated samples.