# Busbar Pair — where the current goes at frequency

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/13_peec/02_busbar_frequency.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

A DC-link busbar is two flat conductors carrying equal and opposite current. At low frequency the
current spreads evenly. Raise the frequency and two things happen at once, both of them
redistribution:

- **skin effect** — current retreats to the surface of each bar;
- **proximity effect** — and, because the return is right there, it crowds onto the **facing** edges,
  where the two currents are closest and the loop encloses least flux.

Both are the same statement: current arranges itself to minimise stored magnetic energy. So the
resistance climbs while the inductance falls — and the inductance **flattens**, because once the
current has reached the facing edges there is nowhere further for it to go.

## An array `freq=` is one solve

```python
sol = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=np.array([1e3, 1e4, 1e5, 1e6, 1e7])).solve()
sol.R, sol.L          # each readout is now an array over the sweep
```

| $f$ | $R$ | $L$ | skin depth |
|---:|---:|---:|---:|
| 1 kHz | 383 µΩ | 36.82 nH | 2.09 mm |
| 10 kHz | 500 µΩ | 33.12 nH | 0.66 mm |
| 100 kHz | 1341 µΩ | 31.63 nH | 0.21 mm |
| 1 MHz | 4947 µΩ | 31.30 nH | 0.066 mm |
| 10 MHz | 16601 µΩ | 31.24 nH | 0.021 mm |

$R$ rises 43×; $L$ falls 15 % and then stops.

## Nothing here was imposed

There is no skin-depth formula in the input and no current profile assumed. The bars are cut into
filaments **across their width**, each is free to carry what it likes, and the redistribution is what
the circuit solves for.

!!! warning "One current per filament"
    The skin effect *within* a filament is not represented — what is captured is redistribution
    *between* them. That is the right model here, because the bars are one cell thick and the
    crowding is across the width. A conductor thick against its own skin depth needs either one cell
    through the thickness (exact — it then carries a current sheet per face) or cells fine enough to
    resolve the profile. `jno.peec` refuses to guess and says so by name.

## Full script

```python
--8<-- "tutorial_examples/13_peec/02_busbar_frequency.py:code"
```
