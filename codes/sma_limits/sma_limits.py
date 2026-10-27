"""
Upper limit on the semi-major axis of an unseen companion from two RV epochs.

Assumptions: primary mass M is known, companion mass m <= M, eccentricity e <= e_max,
inclination and orbital phase are free. Units: au, yr, Msun, so G = 4 pi^2.

Derivation (Newton's 2nd law + Kepler's 2nd law):
    1. The primary feels |dv1/dt| = G m / r^2 (r = separation). Its velocity vector traces
       a circle (the hodograph) of radius G m / h, h = sqrt(G (M+m) a (1-e^2)) being the
       specific angular momentum of the relative orbit. The velocity direction rotates by
       exactly the true anomaly swept, dtheta, so between two epochs
           |Delta v1| = 2 (G m / h) sin(dtheta / 2)          (chord of the hodograph).
    2. The observed RV difference is a projection: Delta v_obs <= |Delta v1|, with equality
       for i = 90 deg and a suitable omega. Any smaller value is reached by lowering sin i.
    3. Kepler's 2nd law (r^2 dtheta/dt = h): dtheta/dt is largest at periastron, so the
       largest dtheta swept in a time Delta t comes from a window centred on periastron,
       and the smallest from one centred on apastron.
    4. An orbit (a, e, m) can produce the data iff
           Delta v_obs <= dv_max(a, e, m) = 2 (G m / h) max_window sin(dtheta / 2).
       a_max is the largest a for which this holds, maximised over e <= e_max and m <= M.
       Both G m / h and dtheta grow with m (larger m -> shorter P), so m = M is the worst
       case; the code still maximises over an m grid to be safe.

Limiting cases (m = M):
    dtheta >= pi reachable (P short enough):  a <= 8 pi^2 M / (Delta v^2 (1-e^2))   [Delta v <= 2K]
    circular, P >> Delta t:                   a <= 2 pi sqrt(M Delta t / Delta v)    [Delta v <= G m Delta t / a^2]
    any orbit, separation between epochs:     r_min <= 2 pi sqrt(M Delta t / Delta v) [Delta v <= G M Delta t / r_min^2]

There is no lower limit from two RVs: short orbits always have K large enough.

Beyond the analytic a_max, likelihood=True maps the likelihood of the measured Delta v in (m, a), marginalised over
inclination, phase, omega, e and the primary mass, and turns it into 1, 2, 3 sigma upper limits on a: from the
profile likelihood (no prior on m or a) and from the posterior with the young close-binary population of
Kounkel et al. (2019) as the period, mass-ratio and eccentricity prior.

The Joker part (needs pymc3 + thejoker, e.g. the `stable` env) samples orbits consistent with the
two RVs for comparison; the analytic a_max is the edge of that distribution.
"""
import os
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

G = 4*np.pi**2  # au^3 / (Msun yr^2)
kms = (1*u.au/u.yr).to(u.km/u.s).value  # km/s per au/yr


def true_anomaly(mean_anomaly, e):
    """True anomaly in [0, pi] for mean anomaly in [0, pi]."""
    E = mean_anomaly + 0.85*e*np.sign(np.sin(mean_anomaly))
    for _ in range(30):
        E = E - (E - e*np.sin(E) - mean_anomaly) / (1 - e*np.cos(E))
    return 2*np.arctan2(np.sqrt(1 + e)*np.sin(E/2), np.sqrt(1 - e)*np.cos(E/2))


def true_anomaly_full(mean_anomaly, e):
    """True anomaly in [0, 2 pi) for any mean anomaly."""
    mean = np.mod(mean_anomaly, 2*np.pi)
    flip = mean > np.pi
    nu = true_anomaly(np.where(flip, 2*np.pi - mean, mean), e)
    return np.where(flip, 2*np.pi - nu, nu)


def dv_max(a, e, m, M, dt):
    """Largest |RV1 - RV2| (au/yr) an orbit (a, e, m) can produce over a time dt (yr)."""
    P = np.sqrt(a**3 / (M + m))
    half = np.pi * (dt % P) / P  # half-width of the window in mean anomaly; full orbits add nothing
    dtheta_max = 2*true_anomaly(half, e)  # window centred on periastron
    dtheta_min = 2*(np.pi - true_anomaly(np.pi - half, e))  # window centred on apastron
    # sin(x/2) peaks at x = pi; every dtheta in [dtheta_min, dtheta_max] is reachable
    S = np.where((dtheta_min <= np.pi) & (np.pi <= dtheta_max), 1., np.maximum(np.sin(dtheta_min/2), np.sin(dtheta_max/2)))
    h = np.sqrt(G*(M + m)*a*(1 - e**2))
    return 2*G*m/h * S


def a_max_grid(dv, dt, M, e_grid, m_grid, a_range=(1e-3, 1e6)):
    """a_max (au) for every (e, m) on the grids; dv in km/s, dt in days. NaN where no orbit works."""
    dv = abs(dv) / kms
    dt = dt / 365.25
    a_grid = np.geomspace(*a_range, 1000)
    e_, m_, a_ = np.meshgrid(e_grid, m_grid, a_grid, indexing='ij')
    ok = dv_max(a_, e_, m_, M, dt) >= dv
    if ok[..., -1].any():
        raise ValueError(f'a_max exceeds {a_range[1]} au; widen a_range.')
    # largest allowed grid point for each (e, m), then bisect to the boundary
    idx = np.minimum(ok.shape[-1] - 1 - np.argmax(ok[..., ::-1], axis=-1), ok.shape[-1] - 2)  # clipped where no a works (NaN below)
    lo, hi = a_grid[idx], a_grid[idx + 1]
    lo = np.where(ok.any(-1), lo, np.nan)
    for _ in range(50):
        mid = np.sqrt(lo*hi)
        good = dv_max(mid, e_grid[:, None], m_grid[None, :], M, dt) >= dv
        lo, hi = np.where(good, mid, lo), np.where(good, hi, mid)
    return lo


def a_max(dv, dt, M, e_max=0.9, n_e=91, n_m=10):
    """
    Largest semi-major axis (au) consistent with an RV difference dv (km/s) over dt (days).
    Returns a_max, the (e, m) that reach it, and a_max(e) maximised over m.
    """
    e_grid = np.linspace(0, e_max, n_e)
    m_grid = np.linspace(M/n_m, M, n_m)
    lo = a_max_grid(dv, dt, M, e_grid, m_grid)
    i, j = np.unravel_index(np.nanargmax(lo), lo.shape)
    return lo[i, j], e_grid[i], m_grid[j], e_grid, np.nanmax(lo, axis=1)


def a_max_with_errors(dv, dv_err, dt, M, M_err, e_max=0.9):
    """
    a_max for every eccentricity cap e <= e_grid[i], with 1 sigma errors propagated from dv_err (km/s) and M_err (Msun).
    a_max falls with dv and rises with M, so each error is the shift in a_max when dv or M moves by 1 sigma,
    taken separately on each side and added in quadrature (asymmetric, linear propagation).
    Returns a_max, the (e, m) that reach it, e_grid, a_cap (a_max for e <= e_grid[i]), and its upper and lower errors.
    """
    a, e, m, e_grid, a_e = a_max(dv, dt, M, e_max)
    cap = lambda dv_i, M_i: np.maximum.accumulate(a_max(dv_i, dt, M_i, e_max)[4]) if dv_i > 0 else np.full_like(a_e, np.inf)
    a_cap = np.maximum.accumulate(a_e)
    up = np.sqrt((cap(dv - dv_err, M) - a_cap)**2 + (cap(dv, M + M_err) - a_cap)**2) if dv_err or M_err else np.zeros_like(a_cap)
    down = np.sqrt((a_cap - cap(dv + dv_err, M))**2 + (a_cap - cap(dv, max(M - M_err, 1e-3)))**2) if dv_err or M_err else np.zeros_like(a_cap)
    return a, e, m, e_grid, a_cap, up, down


def boundary_orbit(a, e, m, M, t, rv):
    """
    Orbit (a, e, m) that reproduces both RVs with the periastron (or apastron) centred between the epochs.
    At a = a_max it needs sin i = 1. Returns RV(t) in km/s, t in days.
    """
    dt = (t[1] - t[0]) / 365.25
    P = np.sqrt(a**3 / (M + m))
    half = np.pi * (dt % P) / P
    dtheta_peri = 2*true_anomaly(half, e)
    dtheta_apo = 2*(np.pi - true_anomaly(np.pi - half, e))
    centre = 0. if np.sin(dtheta_peri/2) >= np.sin(dtheta_apo/2) else np.pi
    M0 = centre - half  # mean anomaly at t[0]
    nu = lambda tt: true_anomaly_full(M0 + 2*np.pi*(np.asarray(tt) - t[0])/365.25/P, e)
    nu1, nu2 = nu(t[0]), nu(t[1])
    # RV = K [cos(nu + omega) + e cos(omega)]; this omega puts the hodograph chord along the line of sight
    omega = np.pi/2 - (nu1 + nu2)/2
    chord = np.cos(nu2 + omega) - np.cos(nu1 + omega)
    if np.sign(chord) != np.sign(rv[1] - rv[0]):
        omega += np.pi
        chord = -chord
    K_max = G*m / np.sqrt(G*(M + m)*a*(1 - e**2)) * kms
    sini = min((rv[1] - rv[0]) / (K_max*chord), 1.)
    K = K_max*sini
    v0 = rv[0] - K*(np.cos(nu1 + omega) + e*np.cos(omega))
    return lambda tt: v0 + K*(np.cos(nu(tt) + omega) + e*np.cos(omega))


def monte_carlo_check(dv, dt, M, e_max=0.9, n=2_000_000, seed=0):
    """Brute force: random orbits with m <= M; largest a among those that can produce dv."""
    rng = np.random.default_rng(seed)
    dv, dt = abs(dv) / kms, dt / 365.25
    m = M*rng.uniform(0, 1, n)
    a = 10**rng.uniform(-2, 3, n)
    e = rng.uniform(0, e_max, n)
    omega = rng.uniform(0, 2*np.pi, n)
    M0 = rng.uniform(0, 2*np.pi, n)
    P = np.sqrt(a**3 / (M + m))
    K = G*m / np.sqrt(G*(M + m)*a*(1 - e**2))  # sin i = 1; lower sin i reaches any smaller dv
    rv = [K*(np.cos(true_anomaly_full(2*np.pi*t/P + M0, e) + omega) + e*np.cos(omega)) for t in [0, dt]]
    feasible = np.abs(rv[1] - rv[0]) >= dv
    return a[feasible].max()


def inclination_averaged_likelihood(A, dv, sigma, n_i=400):
    """
    p(dv | A) with dv = A sin i + noise, cos i uniform in [0, 1] (isotropic orbits); all in the same velocity units.
    With d(cos i) = sin i di, this is the integral over i in [0, pi/2] of N(dv; A sin i, sigma) sin i,
    evaluated only where A sin i is within 8 sigma of dv.
    """
    A = np.atleast_1d(np.abs(A)).astype(float)
    lo = np.arcsin(np.clip((dv - 8*sigma)/A, 0, 1))
    hi = np.arcsin(np.clip((dv + 8*sigma)/A, 0, 1))
    x = np.linspace(0, 1, n_i)
    inc = lo[:, None] + (hi - lo)[:, None]*x[None, :]
    integrand = np.exp(-0.5*((dv - A[:, None]*np.sin(inc))/sigma)**2) / (np.sqrt(2*np.pi)*sigma) * np.sin(inc)
    return np.trapz(integrand, inc, axis=1)


def likelihood_map(dv, dv_err, dt, M, M_err, m_grid, a_grid, e_max=0.9, e_prior='uniform', n_geom=20000, n_P=400, n_M=7, seed=0, return_nodes=False):
    """
    Likelihood of the observed RV difference on a (companion mass, semi-major axis) grid, marginalised over
        - the orbital geometry: isotropic inclination (integrated exactly), uniform omega and phase,
          e from e_prior: 'kounkel' (uniform on [0, min(e_max, kounkel_e_max(P))], Kounkel et al. 2019),
          'uniform' or 'thermal' on [0, e_max], 'circular', or a histogram (bin_edges, counts[, label]), flat
          within each bin and truncated at e_max;
        - the primary mass: M ~ N(M, M_err), by Gauss-Hermite quadrature with n_M nodes, with m <= M at each node.
    No prior on m or a. dv, dv_err in km/s, dt in days, M, M_err, m_grid in Msun, a_grid in au.
    Returns L with shape (len(m_grid), len(a_grid)); with return_nodes also the M nodes, their weights and the
    likelihood at each node (L = sum_k weight_k L_k), which a_posterior needs.
    """
    rng = np.random.default_rng(seed)
    dv, dv_err, dt = abs(dv)/kms, dv_err/kms, dt/365.25
    if M_err > 0:
        x, w = np.polynomial.hermite_e.hermegauss(n_M)
        M_nodes, M_weights = M + M_err*x, w/w.sum()
        M_weights, M_nodes = M_weights[M_nodes > 0], M_nodes[M_nodes > 0]
    else:
        M_nodes, M_weights = np.array([M]), np.array([1.])

    omega = rng.uniform(0, 2*np.pi, n_geom)
    M0 = rng.uniform(0, 2*np.pi, n_geom)
    e = rng.uniform(0, 1, n_geom)  # 'kounkel': scaled by the period-dependent maximum in the loop below
    if not isinstance(e_prior, str):
        edges, counts = np.asarray(e_prior[0], dtype=float), np.asarray(e_prior[1], dtype=float)
        counts = counts*np.clip((e_max - edges[:-1])/np.diff(edges), 0, 1)  # truncate at e_max
        k = rng.choice(len(counts), n_geom, p=counts/counts.sum())
        e = rng.uniform(edges[k], np.minimum(edges[k + 1], e_max))
    elif e_prior == 'kounkel':
        u = e
    elif e_prior == 'uniform':
        e = rng.uniform(0, e_max, n_geom)
    elif e_prior == 'thermal':
        e = e_max*np.sqrt(rng.uniform(0, 1, n_geom))  # f(e) = 2e, truncated at e_max
    elif e_prior == 'circular':
        e = np.zeros(n_geom)
    else:
        raise ValueError(f'Unknown e_prior: {e_prior}')

    # p(dv | A) tabulated once on a log grid of edge-on amplitudes A, then interpolated
    A_table = np.geomspace(dv/50, dv*1e4, 3000)
    L_table = inclination_averaged_likelihood(A_table, dv, dv_err)

    # period grid covering a_grid for every (M node, m)
    M_tot = M_nodes[:, None] + m_grid[None, :]
    P_grid = np.geomspace(np.sqrt(a_grid[0]**3/M_tot.max()), np.sqrt(a_grid[-1]**3/M_tot.min()), n_P)
    half_width = 0.5*np.log(P_grid[1]/P_grid[0])
    # edge-on dv = sqrt(G) m (M+m)^(-2/3) P^(-1/3) D, with D depending only on the geometry and P;
    # histogram |D| per period cell so the M nodes reuse it
    D_edges = np.geomspace(1e-5, 10, 1001)
    D_centres = np.sqrt(D_edges[1:]*D_edges[:-1])
    nu1 = true_anomaly_full(M0, e)
    L_mP = np.zeros((len(M_nodes), len(m_grid), n_P))
    for j, P_centre in enumerate(P_grid):
        # average over the log-P cell rather than its centre: the zeros at P = dt/n (same RV at both epochs) are measure-zero
        P = P_centre*np.exp(rng.uniform(-half_width, half_width, n_geom))
        if e_prior == 'kounkel':
            e = u*np.minimum(e_max, kounkel_e_max(P))
            nu1 = true_anomaly_full(M0, e)
        nu2 = true_anomaly_full(M0 + 2*np.pi*dt/P, e)
        D = np.abs(np.cos(nu2 + omega) - np.cos(nu1 + omega)) / np.sqrt(1 - e**2) * (P/P_centre)**(-1/3)
        weights = np.histogram(np.clip(D, D_edges[0], D_edges[-1]*0.999), D_edges)[0] / n_geom
        used = weights > 0
        scale = np.sqrt(G)*m_grid[None, :]*M_tot**(-2/3)*P_centre**(-1/3)  # (n_M, n_m)
        A = scale[..., None]*D_centres[used]
        L_mP[:, :, j] = np.interp(np.log(A), np.log(A_table), L_table, left=0., right=L_table[-1]) @ weights[used]

    # onto the common a grid (a = ((M+m) P^2)^(1/3) differs per node), then average over M with m <= M at each node
    L_nodes = np.zeros((len(M_nodes), len(m_grid), len(a_grid)))
    for k, M_k in enumerate(M_nodes):
        for i, m in enumerate(m_grid):
            if m <= M_k:
                a_k = ((M_k + m)*P_grid**2)**(1/3)
                L_nodes[k, i] = np.interp(np.log(a_grid), np.log(a_k), L_mP[k, i], left=0., right=0.)
    L = np.tensordot(M_weights, L_nodes, axes=1)
    if return_nodes:
        return L, M_nodes, M_weights, L_nodes
    return L


# Close-binary population of young stars (Kounkel et al. 2019, AJ 157, 196; after Moe & Di Stefano 2017), confirmed
# with APOGEE RVs of 1-100 Myr stars including Orion: log-normal periods, log10(P/d) = 4.6 +- 2.2, P >= 2 d;
# e uniform on [0, e_max(P)], e_max(P) = 1 - (P / 2 d)^(-2/3) <= 0.98 (tidal circularisation);
# q = m/M uniform for m >= 0.08 Msun, plus a twin excess (0.95 < q < 1) holding 20% of systems at a < 0.1 au,
# falling linearly in log a to 0 at a > 200 au.
KOUNKEL_LOG_P, KOUNKEL_P_MIN = (4.6, 2.2), 2/365.25  # log10(P/d); P_min in yr
KOUNKEL_M_MIN, KOUNKEL_TWIN = 0.08, (0.2, 0.1, 200.)  # Msun; twin fraction, a (au) where it is full, a where it is 0

# observed e distribution of solar-type binaries with P < 1000 d (Raghavan et al. 2010, ApJS 190, 1), 34 systems;
# usable as e_prior
RAGHAVAN_E = (np.linspace(0, 1, 11), np.array([4, 5, 5, 6, 2, 7, 2, 1, 1, 1]), 'Raghavan et al. (2010)')


def kounkel_e_max(P):
    """Largest e at period P (yr) in the Kounkel et al. (2019) prior: 1 - (P / 2 d)^(-2/3), between 0 and 0.98."""
    return np.clip(1 - (np.maximum(P, KOUNKEL_P_MIN)/KOUNKEL_P_MIN)**(-2/3), 0, 0.98)


def e_prior_label(e_prior):
    """Name of e_prior for printouts and the figure: a citation for 'kounkel', a histogram's optional third element."""
    if isinstance(e_prior, str):
        return 'Kounkel et al. (2019)' if e_prior == 'kounkel' else e_prior
    return e_prior[2] if len(e_prior) > 2 else 'empirical'


def sigma_levels(n_sigma=(1, 2, 3), df=2):
    """L/L_max at the n-sigma likelihood-ratio contours for df parameters (Wilks: -2 ln(L/L_max) ~ chi^2, df dof)."""
    from scipy.stats import chi2, norm
    return np.exp(-0.5*chi2.ppf(2*norm.cdf(n_sigma) - 1, df=df))


def profile_likelihood_limits(a_grid, L):
    """
    Profile likelihood of a, L_p(a) = max_m L(m, a) / L_max, and its 1, 2, 3 sigma upper limits on a (Wilks, 1 dof).
    No prior on m or a. Returns (L_p, limits).
    """
    profile = L.max(axis=0) / L.max()
    return profile, [a_grid[profile >= level].max() for level in sigma_levels(df=1)]


# log-normal period distribution of solar-type binaries, log10(P/d) = 5.03 +- 2.28 (Raghavan et al. 2010)
RAGHAVAN_LOG_P = (5.03, 2.28)


def posterior_map(m_grid, a_grid, M_nodes, M_weights, L_nodes, prior='kounkel'):
    """
    Joint posterior density in (log10 m, log10 a), unnormalised, from the likelihood_map output (return_nodes=True),
    for a prior on the period and mass ratio q = m/M (at each M node):
        'kounkel':     Kounkel et al. (2019): log-normal P (KOUNKEL_LOG_P) with P >= 2 d; q uniform for
                       m >= 0.08 Msun plus the a-dependent twin excess (KOUNKEL_TWIN);
        'raghavan':    log-normal P of solar-type binaries (RAGHAVAN_LOG_P), q uniform on [0, 1];
        'log-uniform': flat in log P (Opik) over the grid, q uniform. The data give no lower limit on a, so the
                       grid's lower edge (P = 1 d at m = M) is part of this prior and the upper limits move with it.
    """
    log_a = np.log10(a_grid)
    post = np.zeros((len(m_grid), len(a_grid)))
    for M_k, w_k, L_k in zip(M_nodes, M_weights, L_nodes):
        log_P = 1.5*log_a[None, :] - 0.5*np.log10(M_k + m_grid)[:, None] + np.log10(365.25)  # log10(P / d)
        q = m_grid[:, None]/M_k
        if prior == 'kounkel':
            P_density = np.exp(-0.5*((log_P - KOUNKEL_LOG_P[0])/KOUNKEL_LOG_P[1])**2) * (log_P >= np.log10(KOUNKEL_P_MIN*365.25))
            f_twin, a_full, a_zero = KOUNKEL_TWIN
            twin = f_twin*np.clip(np.log10(a_zero/a_grid)/np.log10(a_zero/a_full), 0, 1)[None, :]  # twin fraction at each a
            q_min = min(KOUNKEL_M_MIN/M_k, 0.95)
            q_density = (1 - twin)*((q >= q_min) & (q <= 1))/(1 - q_min) + twin*((q >= 0.95) & (q <= 1))/0.05
        elif prior in ('raghavan', 'log-uniform'):
            mu, sigma = RAGHAVAN_LOG_P
            P_density = np.exp(-0.5*((log_P - mu)/sigma)**2) if prior == 'raghavan' else 1.
            q_density = (q <= 1)*1.
        else:
            raise ValueError(f'Unknown prior: {prior}')
        post += w_k*L_k*q*q_density*P_density  # density in q -> density in log m: times q
    return post


def a_posterior(m_grid, a_grid, M_nodes, M_weights, L_nodes, prior='kounkel'):
    """
    Marginal posterior of a: posterior_map integrated over log m. Returns the density in log10 a, normalised over
    a_grid, and the one-sided upper limits on a at 68.27, 95.45 and 99.73% (1, 2, 3 sigma).
    """
    from scipy.integrate import cumulative_trapezoid
    log_a = np.log10(a_grid)
    pdf = np.trapz(posterior_map(m_grid, a_grid, M_nodes, M_weights, L_nodes, prior), np.log10(m_grid), axis=0)
    pdf /= np.trapz(pdf, log_a)
    cdf = cumulative_trapezoid(pdf, log_a, initial=0.)
    return pdf, 10**np.interp([0.6827, 0.9545, 0.9973], cdf, log_a)


def credible_levels(m_grid, a_grid, density, probs=(0.6827, 0.9545, 0.9973)):
    """Thresholds of a density in (log m, log a) whose highest-density regions enclose probs (1, 2, 3 sigma)."""
    cell = np.gradient(np.log10(m_grid))[:, None]*np.gradient(np.log10(a_grid))[None, :]  # m grid is uneven
    order = np.argsort(density, axis=None)[::-1]
    mass = np.cumsum((density*cell).ravel()[order])
    return density.ravel()[order][np.searchsorted(mass/mass[-1], probs)]


def smooth_map(Z, sigma=1.5):
    """
    Peak-normalised map, lightly smoothed in log: inside the allowed region the likelihood is nearly flat, so
    unsmoothed contours trace Monte Carlo noise.
    """
    from scipy.ndimage import gaussian_filter
    Z = np.exp(gaussian_filter(np.log(np.clip(Z/Z.max(), 1e-12, None)), sigma=sigma))
    return Z / Z.max()


def plot_a_constraints(a_grid, profile, profile_limits, posteriors, resolution_au=None, save_path=None):
    """
    Top: profile likelihood of a with its 1, 2, 3 sigma levels. Bottom: marginal posteriors of a (density in log a)
    for each prior in posteriors = {label: (pdf, limits)}, with their 2 sigma (95.45%) upper limits.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True, gridspec_kw={'hspace': 0.08})
    ax1.plot(a_grid, profile, color='k', linewidth=2)
    for n, (level, a_lim) in enumerate(zip(sigma_levels(df=1), profile_limits), start=1):
        ax1.axhline(level, color='0.45', linestyle=':', linewidth=1)
        ax1.text(a_grid[-1]/1.15, level*1.04, f'{n}$\\sigma$', fontsize=12, color='0.25', ha='right', va='bottom')  # right: the profile is ~0 there
        ax1.plot(a_lim, level, 'o', color='k', markersize=5)
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel('Profile Likelihood', fontsize=15, labelpad=10)

    colors = ['#184f95', '#e07b39', '#5aa469']
    for (label, (pdf, limits)), c in zip(posteriors.items(), colors):
        ax2.plot(a_grid, pdf, color=c, linewidth=2, label=label)
        ax2.axvline(limits[1], color=c, linestyle='--', linewidth=1.5, label=f'95% upper limit: {limits[1]:.1f} au')
    ax2.set_ylim(0, 1.5*max(pdf.max() for pdf, _ in posteriors.values()))  # headroom for the legend
    ax2.set_ylabel(r'$p(\log a)$', fontsize=15, labelpad=10)
    ax2.set_xlabel('Semi-Major Axis (au)', fontsize=15, labelpad=10)
    ax2.legend(loc='upper left', fontsize=12, frameon=False)
    for ax in (ax1, ax2):
        if resolution_au is not None and resolution_au < a_grid[-1]:
            ax.axvspan(resolution_au, a_grid[-1], color='0.5', alpha=0.3, linewidth=0)
        ax.set_xscale('log')
        ax.set_xlim(a_grid[0], a_grid[-1])
        ax.tick_params(axis='both', which='major', labelsize=12)
    if save_path is not None:
        plt.savefig(f'{save_path} - a Constraints.pdf', bbox_inches='tight')
    plt.show()


def place_text(ax, text, candidates, obstacles, boxes, fallback=None, **kwargs):
    """
    Add text at the first candidate (x, y, ha, va), in axes fraction, whose box lies inside the axes and clear of
    the obstacle points (display coordinates) and of earlier labels' boxes. If none is clear: at fallback (e.g. just
    outside the axes) if given, else at the candidate with the fewest conflicts. Appends the chosen box to boxes.
    """
    axes_box = ax.get_window_extent()
    best = None
    for x, y, ha, va in candidates:
        t = ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, **kwargs)
        box = t.get_window_extent().padded(4)
        inside = axes_box.x0 <= box.x0 and box.x1 <= axes_box.x1 and axes_box.y0 <= box.y0 and box.y1 <= axes_box.y1
        cost = (not inside)*1e6 + box.count_contains(obstacles) + 1e3*sum(box.overlaps(b) for b in boxes)
        if best is None or cost < best[0]:
            if best is not None:
                best[1].remove()
            best = (cost, t, box)
        else:
            t.remove()
        if cost == 0:
            break
    if best[0] > 0 and fallback is not None:
        best[1].remove()
        x, y, ha, va = fallback
        t = ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, clip_on=False, **kwargs)
        best = (0, t, t.get_window_extent())
    boxes.append(best[2])
    return best[1]


def plot_ma_map(M, t, rv, m_grid, a_grid, Z, levels, cbar_label, text, e_caps=(0.9,), resolution_au=None, save_file=None):
    """
    Continuous map Z (peak 1) in the (m, a) plane with its 1, 2, 3 sigma contours at levels (descending), the
    analytic a_max(m) for e <= e_cap and, optionally, an imaging resolution limit. Used for both the likelihood
    and the posterior. The resolution label and text are placed where they clear the curves, the contoured region,
    the m = M line and the other labels, so they follow e_caps and the data.
    """
    import matplotlib
    from matplotlib import patheffects

    dt, dv = t[1] - t[0], rv[1] - rv[0]
    m_mesh, a_mesh = np.meshgrid(m_grid, a_grid, indexing='ij')
    contour_color = '#08306b'  # darkest step of the Blues map, with a white halo to stay visible on it
    halo = [patheffects.withStroke(linewidth=2.8, foreground='white')]

    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    pc = ax.pcolormesh(m_mesh, a_mesh, Z, cmap='Blues', vmin=0, vmax=1, shading='gouraud', rasterized=True)
    cs = ax.contour(m_mesh, a_mesh, Z, levels=sorted(levels), colors=contour_color, linewidths=1.)
    for artist in [cs] if isinstance(cs, matplotlib.collections.Collection) else cs.collections:  # matplotlib >= 3.8 or older
        artist.set_path_effects(halo)
    curves = []
    for e_cap, ls in zip(sorted(e_caps, reverse=True), ['-', '--', ':']):
        e_grid = np.linspace(0, e_cap, max(int(round(e_cap/0.01)) + 1, 1))
        m_plot = m_grid[m_grid <= M]
        a_curve = np.nanmax(a_max_grid(dv, dt, M, e_grid, m_plot), axis=0)
        ax.plot(m_plot, a_curve, color='k', linestyle=ls, linewidth=1.8)
        curves.append((e_cap, m_plot, a_curve))
    a_top = a_grid[-1]
    if resolution_au is not None and resolution_au < a_grid[-1]:
        a_top = resolution_au
        ax.axhspan(resolution_au, a_grid[-1], color='0.5', alpha=0.3, linewidth=0)
        ax.axhline(resolution_au, color='0.45', linestyle='-.', linewidth=1.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    if m_grid[-1] > M:
        ax.axvline(M, color='0.45', linestyle=':', linewidth=1.5)
        ax.text(M, 1.01, '$m = M$', transform=ax.get_xaxis_transform(), fontsize=12, color='0.25', ha='center', va='bottom')
    ax.set_xlim(m_grid[0], m_grid[-1])
    ax.set_ylim(a_grid[0], a_grid[-1])
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlabel(r'Companion Mass $\left(M_\odot\right)$', fontsize=15, labelpad=10)
    ax.set_ylabel('Semi-Major Axis (au)', fontsize=15, labelpad=10)
    cbar = fig.colorbar(pc, ax=ax, pad=0.03)
    cbar.set_label(cbar_label, fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    # label each contour on its upper edge (the upper limit), staggered in m so the labels do not collide
    names = dict(zip(levels, [r'1$\sigma$', r'2$\sigma$', r'3$\sigma$']))
    positions = []
    for segments, m_label in zip(cs.allsegs, M*np.array([0.28, 0.4, 0.55])):  # 3, 2, 1 sigma
        points = np.vstack(segments) if len(segments) else np.empty((0, 2))
        near = points[np.abs(np.log(points[:, 0]/m_label)) < 0.1]
        if len(near):
            positions.append(near[np.argmax(near[:, 1])])
    labels = ax.clabel(cs, manual=positions, fmt=names, fontsize=12, inline=True, inline_spacing=4, colors=contour_color)
    for label in labels:
        label.set_path_effects(halo)
    # label each a_max curve along itself, midway through its visible part (after the colorbar fixes the axes shape)
    for e_cap, m_plot, a_curve in curves:
        visible = np.flatnonzero(np.isfinite(a_curve) & (a_curve > a_grid[0]) & (a_curve < a_top))[:-1]
        if len(visible) == 0:
            continue
        j = visible[len(visible)//2]
        p1, p2 = ax.transData.transform([(m_plot[j], a_curve[j]), (m_plot[j + 1], a_curve[j + 1])])
        angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        ax.annotate(f'$e \\leq {e_cap:g}$', (m_plot[j], a_curve[j]), xytext=(-5*np.sin(angle), 5*np.cos(angle)), textcoords='offset points',
                    rotation=np.degrees(angle), rotation_mode='anchor', ha='center', va='bottom', fontsize=12)

    # obstacles for the free labels, in display coordinates: the 3 sigma region, the curves and the m = M line
    to_display = ax.transData.transform
    inside = Z >= min(levels)
    obstacles = [to_display(np.column_stack([m_mesh[inside], a_mesh[inside]]))]
    for _, m_plot, a_curve in curves:
        ok = np.isfinite(a_curve)
        log_m = np.linspace(np.log10(m_plot[ok][0]), np.log10(m_plot[ok][-1]), 500)
        obstacles.append(to_display(np.column_stack([10**log_m, 10**np.interp(log_m, np.log10(m_plot[ok]), np.log10(a_curve[ok]))])))
    a_line = np.geomspace(a_grid[0], a_grid[-1], 300)
    if m_grid[-1] > M:
        obstacles.append(to_display(np.column_stack([np.full_like(a_line, M), a_line])))
    boxes = [label.get_window_extent() for label in ax.texts]
    frac = lambda a: np.log(a/a_grid[0])/np.log(a_grid[-1]/a_grid[0])  # a -> axes fraction
    y_top = 0.97
    if a_top < a_grid[-1]:
        y_res = frac(a_top*1.1)
        place_text(ax, f'Resolution limit: {resolution_au:.1f} au (projected)', [(0.02, y_res, 'left', 'bottom'), (0.98, y_res, 'right', 'bottom'),
                   (0.5, y_res, 'center', 'bottom')], np.vstack(obstacles), boxes, fontsize=12, color='0.25')
        m_band, a_band = np.meshgrid(np.geomspace(m_grid[0], m_grid[-1], 60), np.geomspace(a_top, a_grid[-1], 6))
        obstacles.append(to_display(np.column_stack([m_band.ravel(), a_band.ravel()])))  # keep the text out of the band
        y_top = frac(a_top) - 0.03
    candidates = [(x, y, ha, 'top') for y in np.linspace(y_top, 0.2, 12) for x, ha in [(0.03, 'left'), (0.97, 'right')]]
    place_text(ax, text, candidates, np.vstack(obstacles), boxes, fallback=(0, 1.02, 'left', 'bottom'), fontsize=12, linespacing=1.4)
    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight', dpi=300)
    plt.show()


def companion_mass(K, P, e, M, sini=1.):
    """Solve m sin i / (M+m)^(2/3) = K sqrt(1-e^2) P^(1/3) / (2 pi) for m (K in au/yr, P in yr). NaN if m > M."""
    x = K*np.sqrt(1 - e**2)*P**(1/3) / (2*np.pi) / sini
    lo, hi = np.zeros_like(x), np.full_like(x, M)
    for _ in range(60):  # m / (M+m)^(2/3) increases with m
        mid = (lo + hi)/2
        low = mid/(M + mid)**(2/3) < x
        lo, hi = np.where(low, mid, lo), np.where(low, hi, mid)
    return np.where(x <= M/(2*M)**(2/3), lo, np.nan)


def joker_simulation(M, t, rv, rv_err, e_max=0.9, P_min=1*u.day, P_max=None, sigma_K0=None, n_prior=1_000_000, seed=42):
    """
    Rejection-sample orbits with The Joker, then keep those with m_min (sin i = 1) <= M.
    P_max defaults to twice the period of the analytic a_max orbit, so the prior does not truncate it.
    sigma_K0 defaults to K of an m = M circular companion at P = 1 yr; the Joker K prior scales as
    P^(-1/3) (1-e^2)^(-1/2), the same as a fixed-mass companion, so this tracks m ~ M at all periods.
    Returns the samples and m_min, a(m_min), a(m = M) for each.
    """
    import pymc3 as pm
    import exoplanet.units as xu
    from thejoker import JokerPrior, TheJoker, RVData

    t_rel = np.asarray(t) - t[0]
    data = RVData(t=t_rel, rv=np.asarray(rv)*u.km/u.s, rv_err=np.asarray(rv_err)*u.km/u.s)
    if P_max is None:
        a, _, m, *_ = a_max(rv[1] - rv[0], t_rel[1], M, e_max)
        P_max = 2*np.sqrt(a**3/(M + m))*u.yr
    if sigma_K0 is None:
        sigma_K0 = 2*np.pi*M**(1/3)/2**(2/3)*kms*u.km/u.s

    with pm.Model():
        e = xu.with_unit(pm.Uniform('e', 0, e_max), u.dimensionless_unscaled)
        prior = JokerPrior.default(P_min=P_min, P_max=P_max.to(u.day), sigma_K0=sigma_K0, sigma_v=100*u.km/u.s, pars={'e': e})
    rng = np.random.default_rng(seed)
    joker = TheJoker(prior, random_state=rng)
    samples = joker.rejection_sample(data, prior.sample(size=n_prior, random_state=rng), max_posterior_samples=n_prior)

    P = samples['P'].to(u.yr).value
    K = abs(samples['K'].to(u.au/u.yr).value)
    e = samples['e'].value
    m_min = companion_mass(K, P, e, M)
    keep = np.isfinite(m_min)
    print(f'The Joker: {len(samples)} samples accepted from {n_prior} prior samples (P = {P_min:.0f} to {P_max.to(u.yr):.0f}), {keep.sum()} with m_min <= M.')
    samples, P, m_min = samples[keep], P[keep], m_min[keep]
    return samples, data, m_min, ((M + m_min)*P**2)**(1/3), (2*M*P**2)**(1/3)


def plot_results(M, t, rv, rv_err, samples, data, m_min, a_m_min, a_M, e_caps=(0, 0.5, 0.9), resolution_au=None, n_curves=50, save_path=None, seed=0):
    """(1) a vs. companion mass: Joker samples and analytic a_max(m); (2) RV curves: Joker samples and boundary orbits."""
    from thejoker.plot import plot_rv_curves

    dt = t[1] - t[0]
    dv = rv[1] - rv[0]
    t_rel = np.asarray(t) - t[0]
    m_grid = np.geomspace(M/200, M, 60)
    colors = [f'C{i}' for i in range(len(e_caps))]
    boundaries = []
    for e_cap in e_caps:
        e_grid = np.linspace(0, e_cap, max(int(round(e_cap/0.01)) + 1, 1))
        lo = a_max_grid(dv, dt, M, e_grid, m_grid)
        i = np.nanargmax(lo[:, -1])
        boundaries.append((e_cap, np.nanmax(lo, axis=0), lo[i, -1], e_grid[i]))

    fig, ax = plt.subplots(1, 1, figsize=(6.18, 5))
    sc = ax.scatter(m_min, a_m_min, s=1, c=samples['e'].value, cmap='Greys', vmin=-0.3, vmax=1, rasterized=True, label='The Joker ($m_\\mathrm{min}$, $\\sin i = 1$)')
    for (e_cap, a_m, *_), c in zip(boundaries, colors):
        ax.plot(m_grid, a_m, color=c, linewidth=2, label=f'$a_\\mathrm{{max}}(m)$, $e \\leq {e_cap:g}$')
    # closed-form limiting cases
    dv_auyr, dt_yr = abs(dv)/kms, dt/365.25
    ax.plot(m_grid, 2*np.pi*np.sqrt(m_grid*dt_yr/dv_auyr), color='k', linestyle='--', linewidth=1, label=r'$2\pi\sqrt{m\Delta t/\Delta v}$ ($e = 0$, $P \gg \Delta t$)')
    for (e_cap, *_), c in zip(boundaries, colors):
        ax.plot(m_grid, 16*np.pi**2*m_grid**2/((M + m_grid)*dv_auyr**2*(1 - e_cap**2)), color=c, linestyle=':', linewidth=1.5,
                label=f'$\\Delta v \\leq 2K$, $e = {e_cap:g}$')
    if resolution_au is not None:
        ax.axhline(resolution_au, color='k', linestyle='--', linewidth=1.5, label=f'Resolution limit: {resolution_au:.1f} au (projected)')
    ax.axvline(M, color='k', linewidth=1.5, label=f'$m = M = {M:.2f}~M_\\odot$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(m_grid[0], M*1.15)
    ax.set_ylim(0.5*min(a_m_min.min(), *[np.nanmin(b[1]) for b in boundaries]), 3*max(b[2] for b in boundaries))
    ax.set_xlabel(r'Companion Mass $\left(M_\odot\right)$')
    ax.set_ylabel('Semi-Major Axis (au)')
    cbar = fig.colorbar(sc, ax=ax, label='Eccentricity')
    cbar.ax.set_ylim(0, max(e_caps))
    ax.legend(fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2)
    if save_path is not None:
        plt.savefig(f'{save_path} - Allowed Param.pdf', bbox_inches='tight')
    plt.show()

    # top: data window with Joker samples; bottom: boundary orbits over a full period of the widest one
    fig, (ax, ax_wide) = plt.subplots(2, 1, figsize=(7, 7.5))
    t_grid = np.linspace(-dt, 2*dt, 4096)
    rng = np.random.default_rng(seed)
    long_P = np.flatnonzero(samples['P'].to(u.day).value > dt/3)
    idx = rng.choice(long_P, size=min(n_curves, len(long_P)), replace=False)
    plot_rv_curves(samples[idx], t_grid, rv_unit=u.km/u.s, data=data, ax=ax,
                   plot_kwargs=dict(color='C7', linewidth=0.4, alpha=0.6), data_plot_kwargs=dict(color='k', marker='o', markersize=5, elinewidth=1.2, capsize=3, capthick=1.2, zorder=10))
    P_wide = np.sqrt(boundaries[-1][2]**3 / (2*M))*365.25  # days
    t_wide = dt/2 + np.linspace(-0.6, 0.6, 20000)*P_wide
    for (e_cap, _, a, e), c in zip(boundaries, colors):
        orbit = boundary_orbit(a, e, M, M, t_rel, rv)
        label = f'$a_\\mathrm{{max}}$ orbit, $e \\leq {e_cap:g}$: $a = {a:.1f}$ au, $e = {e:.2f}$, $P = {np.sqrt(a**3/(2*M)):.1f}$ yr'
        ax.plot(t_grid, orbit(t_grid), color=c, linewidth=1.5, label=label)
        ax_wide.plot(t_wide/365.25, orbit(t_wide), color=c, linewidth=1, label=label)
    ax.plot([], [], color='C7', linewidth=0.8, label=f'The Joker ({len(idx)} samples, $P > \\Delta t/3$)')
    ax.set_ylim(min(rv) - 3*abs(dv), max(rv) + 3*abs(dv))
    ax.set_xlabel(f'Days after first epoch')
    ax.set_ylabel(r'RV ($\mathrm{km}\cdot\mathrm{s}^{-1}$)')
    ax.legend(fontsize=7, loc='upper right')
    ax_wide.errorbar(t_rel/365.25, rv, yerr=rv_err, color='k', fmt='o', markersize=4, zorder=10)
    ax_wide.set_xlabel('Years after first epoch')
    ax_wide.set_ylabel(r'RV ($\mathrm{km}\cdot\mathrm{s}^{-1}$)')
    ax_wide.legend(fontsize=7, loc='upper right')
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(f'{save_path} - Orbits.pdf', bbox_inches='tight')
    plt.show()


def sma_limits_from_two_rvs(M, t, rv, M_err=0., rv_err=(0., 0.), jitter=0., e_max=0.9, check=False, joker=False, likelihood=False, e_prior='kounkel', prior='kounkel', resolution_au=None, save_dir=None):
    """
    Semi-major axis limits for an unseen companion (m <= M) from two RV epochs.

    Required:
        M: primary mass (Msun)
        t: the two epochs (days; MJD, JD, ... -- only the difference is used)
        rv: the two RVs (km/s), same order as t
    Strongly recommended:
        rv_err: RV uncertainties (km/s). Needed for the a_max errors, the likelihood map and The Joker.
        M_err: primary mass uncertainty (Msun), propagated into the a_max errors and marginalised over in the likelihood map
            (Gaussian in M, with m <= M at each M; the map extends to m = M + 2 M_err).
        jitter: extra RV noise (km/s) added in quadrature to each RV error, everywhere errors are used
            (significance of dv, likelihood map, The Joker). ~0.5-1 km/s for young spotted stars; should also
            cover any zero-point offset between the two instruments.
    Options:
        e_max: assumed maximum eccentricity. a_max for e <= e_max depends strongly on it (Parenago 1837: 3.3 au
            for e = 0, 31 au for e <= 0.9).
        likelihood: map the geometry-marginalised likelihood in (m, a) with 1, 2, 3 sigma contours (~40 s).
            This is where the RV errors and jitter enter. Also gives 1D upper limits on a from the profile
            likelihood (no prior on m or a) and from the posterior, and the 2D posterior map.
        e_prior: eccentricity prior of the likelihood (and so of the posterior): 'kounkel' (uniform up to the
            period-dependent maximum of Kounkel et al. 2019, a tidal-circularisation envelope, further capped at
            e_max), 'uniform', 'thermal' (f(e) = 2e on [0, e_max]), 'circular', or an observed histogram
            (bin_edges, counts[, label]) such as RAGHAVAN_E. It shapes the likelihood, not the a_max curve.
        prior: period and mass-ratio prior of the posterior: 'kounkel' (young close binaries, Kounkel et al. 2019:
            log-normal P, P >= 2 d, uniform q for m >= 0.08 Msun plus twins), 'raghavan' or 'log-uniform' (see
            posterior_map). The other two are printed for comparison either way.
        resolution_au: imaging resolution limit to draw, in projected au = angle (arcsec) x distance (pc).
        save_dir: directory for the figures (created if needed), named by the inputs, e.g.
            'M0.528_dv7.80_dt330_Kounkel - Likelihood Map.pdf'; None only shows the figures.
        joker: run The Joker and plot its samples and the a_max orbits (~50 s; needs pymc3 + thejoker, e.g. the `stable` env).
        check: brute-force Monte Carlo check that no random orbit exceeds the analytic a_max (slow).

    a_max is the largest a satisfying dv <= 2 (G m / h) max sin(dtheta / 2) (module docstring) at the measured dv
    and M, for m <= M and e <= e_max. Its 1 sigma errors come from propagating sigma_dv (incl. jitter) and M_err.
    Returns a dict: 'a_max' (au, e <= e_max) and 'a_max_err' (upper, lower); plus 'joker' (samples), 'likelihood' (m_grid, a, L)
    and 'a_upper' (1, 2, 3 sigma upper limits on a: profile likelihood and each posterior) if those were run.
    Everything except joker=True needs only numpy, scipy, astropy and matplotlib.
    """
    rv_err = np.sqrt(np.asarray(rv_err, dtype=float)**2 + jitter**2)  # jitter in quadrature, used everywhere below
    dt = abs(t[1] - t[0])
    dv = abs(rv[1] - rv[0])
    save_path = None
    if save_dir is not None:
        save_dir = os.path.expanduser(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'M{M:.3f}_dv{dv:.2f}_dt{dt:.0f}_{e_prior_label(e_prior).split()[0]}')
    dv_err = np.hypot(*rv_err)
    significance = dv/dv_err if dv_err else np.inf
    print(f'Δt = {dt:.1f} d, Δv = {dv:.2f} ± {dv_err:.2f} km/s ({significance:.1f}σ, jitter = {jitter:g} km/s), M = {M:.3f} ± {M_err:.3f} Msun')
    if significance < 3:
        print(f'  Warning: Δv is only {significance:.1f}σ; the RV variation may not be real.')

    a, e, m, e_grid, a_cap, a_up, a_down = a_max_with_errors(dv, dv_err, dt, M, M_err, e_max)
    results = {'a_max': a, 'a_max_err': (a_up[-1], a_down[-1])}
    print(f'\na_max (e <= {e_max:g}) = {a:.2f} +{a_up[-1]:.2f} -{a_down[-1]:.2f} au  [reached at e = {e:.2f}, m = {m:.3f} Msun, P = {np.sqrt(a**3/(M + m)):.1f} yr]')
    for e_i in [0.7, 0.5, 0.3]:
        if e_i < e_max:
            i = np.searchsorted(e_grid, e_i + 1e-9) - 1
            print(f'a_max (e <= {e_i:g}) = {a_cap[i]:.2f} +{a_up[i]:.2f} -{a_down[i]:.2f} au')
    print(f'a_max (e = 0)   = {a_cap[0]:.2f} +{a_up[0]:.2f} -{a_down[0]:.2f} au')
    r_min = 2*np.pi*np.sqrt(M*dt/365.25/(dv/kms))
    r_min_err = r_min/2*np.hypot(M_err/M, dv_err/dv)
    print(f'r_min between epochs <= 2π√(MΔt/Δv) = {r_min:.2f} ± {r_min_err:.2f} au (any e; also the long-period e = 0 a_max)')
    print(f'Δv <= 2K bound (e = {e_max:g})       = {8*np.pi**2*M/((dv/kms)**2*(1 - e_max**2)):.3f} au')
    if check:
        print(f'Monte Carlo max feasible a = {monte_carlo_check(dv, dt, M, e_max):.3f} au (should be <= a_max)')

    if joker:
        print()
        samples, data, m_min, a_m_min, a_M = joker_simulation(M, t, rv, rv_err, e_max)
        print(f'  Joker a (m = M): median {np.median(a_M):.3f} au, 95% {np.percentile(a_M, 95):.3f} au, 99% {np.percentile(a_M, 99):.3f} au, max {a_M.max():.3f} au (analytic a_max {a:.3f} au)')
        results['joker'] = samples
        e_caps = sorted({0, 0.5, e_max} if e_max >= 0.5 else {0, e_max})
        plot_results(M, t, rv, rv_err, samples, data, m_min, a_m_min, a_M, e_caps=e_caps, resolution_au=resolution_au, save_path=save_path)

    if likelihood:
        # m up to M + 2 M_err: a companion heavier than the nominal M is allowed if the true M is larger
        m_grid = np.geomspace(M/200, M, 60)
        if M_err > 0:
            m_grid = np.concatenate([m_grid, np.linspace(M, M + 2*M_err, 9)[1:]])
        i_M = len(np.geomspace(M/200, M, 60)) - 1  # row of the nominal M
        a_grid = np.geomspace(((2*M)/365.25**2)**(1/3), 2**(2/3)*a, 300)  # P = 1 d at m = M up to twice the a_max period
        L, M_nodes, M_weights, L_nodes = likelihood_map(dv, dv_err, dt, M, M_err, m_grid, a_grid, e_max, e_prior, return_nodes=True)
        ratio = L / L.max()
        print(f'\nLikelihood map ({e_prior_label(e_prior)} e <= {e_max:g}, σ_Δv = {dv_err:.2f} km/s, M = {M:.3f} ± {M_err:.3f} Msun); a = {a_grid[0]:.3f} to {a_grid[-1]:.0f} au:')
        a_mesh = np.broadcast_to(a_grid, L.shape)
        for n, level in zip([1, 2, 3], sigma_levels()):
            inside = ratio >= level
            edge = ' (lower end is the grid edge: no lower limit)' if inside[:, :3].any() else ''
            at_M = f'at m = M: {a_grid[inside[i_M]].min():.3f} to {a_grid[inside[i_M]].max():.3f} au' if inside[i_M].any() else 'm = M outside'
            print(f'  {n}σ: a = {a_mesh[inside].min():.3f} to {a_mesh[inside].max():.3f} au over all m{edge};  {at_M}')
        results['likelihood'] = (m_grid, a_grid, L)
        e_text = f'$e$ prior: {e_prior_label(e_prior)}'
        plot_ma_map(M, t, rv, m_grid, a_grid, smooth_map(L), sigma_levels(), r'$\mathcal{L} / \mathcal{L}_\mathrm{max}$', e_text,
                    e_caps=(e_max,), resolution_au=resolution_au, save_file=save_path and f'{save_path} - Likelihood Map.pdf')

        # 1D upper limits on a: profile likelihood (no prior on m, a) and posteriors
        prior_labels = {'kounkel': 'Kounkel et al. (2019)', 'raghavan': 'Raghavan et al. (2010) $P$, uniform $q$',
                        'log-uniform': 'log-uniform $P$ ($\\geq$ 1 d), uniform $q$'}
        profile, profile_limits = profile_likelihood_limits(a_grid, L)
        print('\nUpper limits on a (au)                       1σ       2σ       3σ')
        print('  profile likelihood (no m, a prior)    ' + ''.join(f'{x:9.3f}' for x in profile_limits))
        posteriors = {}
        for name in [prior] + [p for p in prior_labels if p != prior]:
            pdf, limits = a_posterior(m_grid, a_grid, M_nodes, M_weights, L_nodes, name)
            posteriors[name] = (pdf, limits)
            print(f'  posterior, {name + " P and q prior":26s} ' + ''.join(f'{x:9.3f}' for x in limits) + ('' if name == prior else '  (comparison)'))
        print('  (posterior limits are one-sided: 68.27, 95.45, 99.73% of the posterior below them)')
        results['a_upper'] = {'profile': profile_limits, **{k: v[1] for k, v in posteriors.items()}}
        plot_a_constraints(a_grid, profile, profile_limits, {f'Posterior, {prior_labels[prior]} priors': posteriors[prior]},
                           resolution_au=resolution_au, save_path=save_path)

        # joint posterior, highest-density 1, 2, 3 sigma regions
        post = smooth_map(posterior_map(m_grid, a_grid, M_nodes, M_weights, L_nodes, prior))
        text = f'Priors ($e$, $P$, $q$): {prior_labels[prior]}' if e_prior == prior else f'{e_text}\n$P$, $q$ prior: {prior_labels[prior]}'
        plot_ma_map(M, t, rv, m_grid, a_grid, post, credible_levels(m_grid, a_grid, post), 'Posterior density (peak = 1)', text,
                    e_caps=(e_max,), resolution_au=resolution_au, save_file=save_path and f'{save_path} - Posterior Map.pdf')
    return results


if __name__ == '__main__':
    # Parenago 1837 (HC2000 546): APOGEE 2019-02-25, NIRSPEC 2020-01-21
    sma_limits_from_two_rvs(
        M=0.52814,               # primary mass (Msun)                                    REQUIRED
        M_err=0.05,              # primary mass uncertainty (Msun); placeholder -- use mass_e_MIST
        t=[58539, 58869],        # the two epochs, in days (MJD, JD, ...)                 REQUIRED
        rv=[29.14, 21.34],       # the two RVs (km/s), same order as t                     REQUIRED
        rv_err=[0.38, 0.28],     # RV uncertainties (km/s)                                 strongly recommended
        jitter=0.,               # extra RV noise (km/s) per RV, in quadrature; ~0.5-1 for young stars
        e_max=0.9,               # assumed maximum eccentricity; a_max depends strongly on it
        check=False,             # brute-force Monte Carlo check of a_max (slow)
        joker=False,             # The Joker samples + orbit plots (~50 s; needs the `stable` env)
        likelihood=True,         # likelihood and posterior maps, 1D upper limits on a (~1 min)
        e_prior='kounkel',       # e prior: 'kounkel', 'uniform', 'thermal', 'circular', RAGHAVAN_E or (bin_edges, counts[, label])
        prior='kounkel',         # posterior P and q prior: 'kounkel', 'raghavan' or 'log-uniform'
        resolution_au=0.05*389,  # imaging limit = angle (arcsec) x distance (pc); NIRSPAO K-band AO, ~0.05" at 389 pc
        save_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')  # None only shows them
    )
