import equinox as eqx
import diffrax as dfx
import jax
import jax.flatten_util as jfu
import jax.lax as lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optimistix as optx
from collections.abc import Callable
from equinox.internal import ω
from jaxtyping import Array, Float
from mpl_toolkits.mplot3d import Axes3D
from typing import TypeAlias


jax.config.update("jax_enable_x64", True)


# M = 4^N - 1 for a system of N qubits.
Scalar: TypeAlias = Float[Array, ""]
ArrayM: TypeAlias = Float[Array, "M"]
ArrayMMM: TypeAlias = Float[Array, "M M M"]


def _vector_field(
    t: Scalar, y: tuple[ArrayM, ArrayM], Γ: Callable[[ArrayM], ArrayMMM]
) -> tuple[ArrayM, ArrayM]:
    x, v = y
    dx = v
    dv = -jnp.einsum("IJK,J,K->I", Γ(x), v, v)
    return dx, dv


def _solve_ivp(
    t0: Scalar,
    t1: Scalar,
    x0: ArrayM,
    v0: ArrayM,
    Γ: Callable[[ArrayM], ArrayMMM],
    saveat: dfx.SaveAt,
) -> dfx.Solution:

    # Handle singularities: convert inf to max, and -inf to min.
    # Then 0 * inf produces 0, not NaN.
    def Γ_clipped(x):
        info = jnp.finfo(jnp.result_type(x))
        return jnp.clip(Γ(x), a_min=info.min, a_max=info.max)

    term = dfx.ODETerm(_vector_field)
    solver = dfx.Tsit5()
    dt0 = None
    y0 = (x0, v0)
    stepsize_controller = dfx.PIDController(
        rtol=1e-8, atol=1e-8, pcoeff=0.4, icoeff=0.3
    )
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        y0,
        args=Γ_clipped,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
        throw=False,
    )
    # TODO: raise error with JAX
    lax.cond(eqx.internal.unvmap_all(sol.result == dfx.RESULTS.successful), lambda: None, lambda: jax.debug.breakpoint())
    return sol


def solve_ivp_t1(
    t0: Scalar, t1: Scalar, x0: ArrayM, v0: ArrayM, Γ: Callable[[ArrayM], ArrayMMM]
) -> tuple[ArrayM, ArrayM]:
    saveat = dfx.SaveAt(t1=True)
    sol = _solve_ivp(t0, t1, x0, v0, Γ, saveat)
    (xs, vs) = sol.ys
    (x1,) = xs
    (v1,) = vs
    return x1, v1


@eqx.filter_jit
def solve_ivp_dense(
    t0: Scalar, t1: Scalar, x0: ArrayM, v0: ArrayM, Γ: Callable[[ArrayM], ArrayMMM]
) -> Callable[[Scalar], tuple[ArrayM, ArrayM]]:
    saveat = dfx.SaveAt(dense=True)
    sol = _solve_ivp(t0, t1, x0, v0, Γ, saveat)
    # Drop the velocity component
    return eqx.Partial(lambda f, x: f(x)[0], sol.evaluate)


class _ClippedGradient(optx.AbstractDescent):
    def __call__(self, step_size, args, options):
        vector = options["vector"]
        diff = (vector**ω / optx.max_norm(vector)).ω
        return (-step_size * diff**ω).ω, optx.RESULTS.successful


class _ClippedGradientDescent(optx.AbstractGradientDescent):
    def __init__(self, rtol: float, atol: float, learning_rate: float):
        self.rtol = rtol
        self.atol = atol
        self.norm = optx.max_norm
        self.line_search = optx.LearningRate(_ClippedGradient(), learning_rate)


@eqx.filter_jit
def solve_bvp(
    target: ArrayM,
    pieces: int,
    Γ: Callable[[ArrayM], ArrayMMM],
    canonicalise: None | Callable[[ArrayM], ArrayM] = None,
) -> ArrayM:
    ts = jnp.linspace(0, 1, pieces + 1)
    t0s = ts[:-1]
    t1s = ts[1:]
    (M,) = target.shape
    init_later_x0s = jax.vmap(
        lambda x: jnp.linspace(0, x, pieces + 1, dtype=target.dtype)[1:-1], out_axes=1
    )(target)
    # TODO: initialise better
    init_v0s = jnp.zeros((pieces, M), target.dtype)
    if canonicalise is not None:
        target = canonicalise(target)

    def _root_fn(later_x0s: Float[Array, "M-1"], v0s: ArrayM):  # 2M - 1 inputs
        zero = jnp.zeros((1, M), target.dtype)
        x0s = jnp.concatenate([zero, later_x0s])
        (x1s, v1s) = eqx.filter_vmap(solve_ivp_t1)(t0s, t1s, x0s, v0s, Γ)
        if canonicalise is not None:
            x0s = jax.vmap(canonicalise)(x0s)
            x1s = jax.vmap(canonicalise)(x1s)
        x_diff = x0s[1:] - x1s[:-1]  # M - 1 conditions
        v_diff = v0s[1:] - v1s[:-1]  # M - 1 conditions
        target_diff = x1s[-1] - target  # 1 condition
        return x_diff, v_diff, target_diff  # 2M - 1 outputs

    def _min_fn(z0, args):
        later_x0s, v0s = z0
        out, _ = jfu.ravel_pytree(_root_fn(later_x0s, v0s))
        loss = jnp.mean(out**2)
        return loss

    # TODO: substitute for OptaxMinimiser?
    solver = _ClippedGradientDescent(rtol=1e-8, atol=1e-8, learning_rate=1e-2)
    sol = optx.minimise(_min_fn, solver, (init_later_x0s, init_v0s), max_steps=1000)
    later_x0s, v0s = sol.value
    return v0s


def demo_sphere():
    # θ_target = 2.1
    # φ_target = 0.8
    θ_target = 0.3
    φ_target = 0.7
    target = jnp.array([θ_target, φ_target])
    pieces = 4

    def Γ(x: Float[Array, "2"]) -> Float[Array, "2 2 2"]:
        θ, _ = x
        Γ⁀θ_θθ = 0
        Γ⁀θ_θφ = 0
        Γ⁀θ_φθ = 0
        Γ⁀θ_φφ = -jnp.cos(θ) * jnp.sin(θ)
        Γ⁀φ_θθ = 0
        Γ⁀φ_θφ = 1 / jnp.tan(θ)
        Γ⁀φ_φθ = 1 / jnp.tan(θ)
        Γ⁀φ_φφ = 0
        return jnp.array(
            [[[Γ⁀θ_θθ, Γ⁀θ_θφ], [Γ⁀θ_φθ, Γ⁀θ_φφ]], [[Γ⁀φ_θθ, Γ⁀φ_θφ], [Γ⁀φ_φθ, Γ⁀φ_φφ]]]
        )

    def canonicalise(x: Float[Array, "2"]) -> Float[Array, "2"]:
        # θ is the angle from the vertical
        # φ is the angle from the horizontal
        θ, φ = x
        θ = θ % (2 * jnp.pi)
        pred = θ < jnp.pi
        θ = jnp.where(pred, θ, 2 * jnp.pi - θ)
        φ = jnp.where(pred, φ, jnp.pi + φ)
        φ = φ % (2 * jnp.pi)
        return jnp.stack([θ, φ])

    v0 = solve_bvp(target, pieces, Γ, canonicalise)
    sol = solve_ivp_dense(0, 1, jnp.zeros_like(target), v0, Γ)
    θs, φs = jax.vmap(sol, out_axes=1)(jnp.linspace(0, 1, 100))

    # θs = jnp.linspace(0, θ_target, pieces + 1, dtype=target.dtype)
    # φs = jnp.linspace(0, φ_target, pieces + 1, dtype=target.dtype)

    def to_3d(θ, φ):
        x = jnp.sin(θ) * jnp.cos(φ)
        y = jnp.sin(θ) * jnp.sin(φ)
        z = jnp.cos(θ)
        return x, y, z

    θ_surface, φ_surface = np.mgrid[0.0 : np.pi : 100j, 0.0 : 2 * np.pi : 100j]
    x_surface, y_surface, z_surface = to_3d(θ_surface, φ_surface)
    x_path, y_path, z_path = to_3d(θs, φs)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x_surface, y_surface, z_surface, rstride=1, cstride=1, color="c", alpha=0.7, linewidth=0)
    ax.plot(1.01 * x_path, 1.01 * y_path, 1.01 * z_path, color="r", linewidth=3)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    demo_sphere()
