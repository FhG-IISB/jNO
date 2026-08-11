# --8<-- [start:code]
"""16 - Thermal barrier coating: two bodies, two mesh resolutions, one solve (``conforming=False``).

A thin ceramic coating (thickness 0.05, k = 1) on a thick metal substrate (thickness 1.0, k = 20).
This is the geometry of a thermal barrier coating, and its point is that a layer **20x thinner** made of
a material **20x less conductive** carries the *same* temperature drop as the whole substrate beneath
it: conduction in series adds thermal resistances L/k, and here both layers contribute 0.05.

With a steady flux q injected at the top and T = 0 held at the base, the exact 1-D solution is
piecewise linear with a kink at the interface::

    T_interface = q L_sub / k_sub                      = 1 * 1.00 / 20 = 0.050
    T_top       = T_interface + q L_film / k_film      + 1 * 0.05 /  1 = 0.100

so the coating -- 5 % of the thickness -- is responsible for half the temperature rise. That is what
makes it a *barrier*, and it is also what makes it hard to mesh: all the action is inside a sliver.

**Why two meshes.** Resolving the gradient through the coating needs several elements across 0.05, so
h ~ 0.0125; the substrate is happy at h ~ 0.12, ten times coarser. One conforming mesh cannot honour
both at the shared interface -- it has to pick a single size there and compromise. Measured on this
geometry, fragmenting the two bodies into one mesh gives:

    conforming      coating h = 0.0227 (asked 0.0125)   substrate h = 0.0995 (asked 0.12)
    conforming=False coating h = 0.0119 (asked 0.0125)  substrate h = 0.1084 (asked 0.12)

i.e. it under-resolves the layer you care about *and* over-refines the bulk you do not. Meshing each
body independently and gluing them with a tie ``u(A) - u(B)`` gets both right. The two interface
surfaces then carry different node layouts, so ``jno.fem`` couples them with a **mortar** (integrated)
constraint rather than node-to-node matching -- see ``docs/fem.md``, "Tying two boundaries".

Verified against the exact series-resistance solution above, at the interface and at the top surface.
"""

import numpy as np

import jno

K_SUB, K_FILM = 20.0, 1.0  # metal substrate / ceramic coating conductivity
L_SUB, L_FILM = 1.0, 0.05  # thicknesses
Q = 1.0  # heat flux injected at the top surface
H_SUB, H_FILM = 0.12, 0.0125  # per-region mesh sizes: the coating needs ~4 elements through thickness

# Two bodies, each meshed at its OWN size. `conforming=False` skips the fragment, so the shared
# surface exists twice -- once per body -- and each is meshed independently.
d = jno.Shape.regions(
    substrate=jno.Shape.rect(0.0, 0.0, 1.0, L_SUB, size=H_SUB),
    coating=jno.Shape.rect(0.0, L_SUB, 1.0, L_SUB + L_FILM, size=H_FILM),
    conforming=False,
).domain()
# `Shape` already auto-names each body's edges, and with the two bodies stacked the outer `bottom`
# (y = 0, on the substrate) and `top` (y = 1.05, on the coating) are exactly the two surfaces wanted.

T, phi = d.fem_symbols()
# One conduction term per material region -- each integrates over that region's cells only. (The
# `d.by_region({...})` shorthand is for regions declared as geometry parts or `d.tag` predicates; a
# `Shape.regions` name is not one of those, so with two materials the explicit form is also the
# clearer one.)
xs, ys, _ = d.variable("substrate", split=True)
xc, yc, _ = d.variable("coating", split=True)
Ts, ps = T.bind(x=xs, y=ys), phi.bind(x=xs, y=ys)
Tc, pc = T.bind(x=xc, y=yc), phi.bind(x=xc, y=yc)

# The two sides of the interface. The shared surface exists TWICE -- once per body, at identical
# coordinates -- so a spatial predicate selects both; `region=` names which body owns the facets, and
# is the only thing that can tell them apart.
#
# ORDER MATTERS: in `u(A) - u(B)` the first is the SECONDARY, whose interface DOFs are eliminated in favour
# of an interpolation from the main, so the secondary must be the *finer* side or the fine mesh's
# resolution at the interface is discarded. Measured here:
#     secondary = coating   (81 interface nodes)  ->  T_interface = 0.05000   exact
#     secondary = substrate (10 interface nodes)  ->  T_interface = 0.05531   10.62% error
on_interface = lambda x, y: np.abs(y - L_SUB) < 1e-9  # noqa: E731
a = d.variable("film_face", where=on_interface, region="coating", split=True)  # secondary: the finer side
b = d.variable("base_face", where=on_interface, region="substrate", split=True)  # main

xt, yt, _ = d.variable("top", split=True)  # coating outer surface: flux q in
xb, yb, _ = d.variable("bottom", split=True)  # substrate base: T = 0

fem = jno.fem(
    [
        K_SUB * (Ts.x * ps.x + Ts.y * ps.y),  # steady conduction in the metal substrate
        K_FILM * (Tc.x * pc.x + Tc.y * pc.y),  # ... and in the ceramic coating
        -Q * phi.bind(x=xt, y=yt),  # flux in at the top (a natural BC)
        T(*a) - T(*b),  # glue the two bodies
        T(xb, yb) - 0.0,  # T = 0 at the base
    ]
)
sol = np.asarray(fem.solve()).reshape(-1)
pts = np.asarray(fem.points)

exact_iface = Q * L_SUB / K_SUB
exact_top = exact_iface + Q * L_FILM / K_FILM
# 1e-6, not 1e-9: `fem.points` is float32 by default, whose resolution at y ~ 1 is ~1.2e-7, so a
# 1e-9 band selects nothing at all and the mean comes back NaN.
got_iface = float(sol[np.abs(pts[:, 1] - L_SUB) < 1e-6].mean())
got_top = float(sol[pts[:, 1] > L_SUB + L_FILM - 1e-6].mean())

print(
    f"\nCoating (tied, two mesh resolutions): dofs={fem.dofs}\n"
    f"  T at interface  FEM={got_iface:.5f}  exact={exact_iface:.5f}\n"
    f"  T at top        FEM={got_top:.5f}  exact={exact_top:.5f}\n"
    f"  the 0.05-thick coating carries {100 * (got_top - got_iface) / got_top:.0f}% of the temperature rise"
)
assert abs(got_iface - exact_iface) / exact_top < 0.02
assert abs(got_top - exact_top) / exact_top < 0.02
# --8<-- [end:code]
