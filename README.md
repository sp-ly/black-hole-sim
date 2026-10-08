# black-hole-sim

for my astronomy class (as well as a bit of a passion project). sim runs in matplotlib

schwarzchild_simulator.py -> we test a particle under effective potential $V_{\mathrm{eff}}(r) = (1 - 2M/r)(1 + L^2/r^2)$ with conserved energy $E$ and angular momentum $L$. we define the black hole through the schwarzchild metric, $r_s = 2GM/c^2$

kerr_simulator.py -> we kinda do the same here but incorporate angular momentum $a$ and frame dragging. we define the black hole through a simplified version of the kerr metric,

$$ ds^2 = -\left(1-\frac{2Mr}{\rho^2}\right)dt^2
-\frac{4Mar\sin^2\theta}{\rho^2}\,dt\,d\phi
+\frac{\Sigma}{\rho^2}\sin^2\theta\,d\phi^2
+\frac{\rho^2}{\Delta}dr^2
+\rho^2 d\theta^2
$$

<p align="center">with</p>

$$
\rho^2 = r^2 + a^2 \cos^2\theta,
\quad
\Delta = r^2 - 2Mr + a^2,
\quad
\Sigma = (r^2+a^2)^2 - a^2\Delta\sin^2\theta.
$$
