import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import lineax as lx
import matplotlib.pyplot as plt
import numpy as np
import operator
import optax
import optimistix as optx
from collections.abc import Callable
from jaxtyping import Array, Complex, Float, Int
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from numpy import ndarray
from typing import TypeAlias, Union


jax.config.update("jax_enable_x64", True)


# M = 4^N - 1 for a system of N qubits.
FloatScalar: TypeAlias = Float[Array, ""]
Array1D: TypeAlias = Float[Array, " _"]
ArrayM: TypeAlias = Float[Array, " M"]
ArrayMM: TypeAlias = Float[Array, "M M"]
ArrayMMM: TypeAlias = Float[Array, "M M M"]


def _vector_field(
    t: FloatScalar, y: tuple[ArrayM, ArrayM], christoffel: Callable[[ArrayM], ArrayMMM]
) -> tuple[ArrayM, ArrayM]:
    x, v = y
    dx = v
    v = eqx.internal.error_if(v, jnp.abs(jnp.max(v)) > 1e4, "oh no")
    dv = -jnp.einsum("IJK,J,K->I", christoffel(x), v, v)
    return dx, dv


def _solve_single_ivp(
    t0: FloatScalar,
    t1: FloatScalar,
    x0: ArrayM,
    v0: ArrayM,
    christoffel: Callable[[ArrayM], ArrayMMM],
    diffeq_solver: dfx.AbstractSolver,
    max_steps: None | int,
) -> tuple[ArrayM, ArrayM]:
    term = dfx.ODETerm(_vector_field)
    dt0 = None
    y0 = (x0, v0)
    stepsize_controller = dfx.PIDController(
        rtol=1e-6, atol=1e-6, pcoeff=0.3, icoeff=0.4
    )
    sol = dfx.diffeqsolve(
        term,
        diffeq_solver,
        t0,
        t1,
        dt0,
        y0,
        args=christoffel,
        stepsize_controller=stepsize_controller,
        adjoint=dfx.DirectAdjoint(),  # TODO
        max_steps=max_steps,
    )
    (xs, vs) = sol.ys
    (x1,) = xs
    (v1,) = vs
    return x1, v1


def _vector_field_with_length(t, y, args):
    (x, v), _ = y
    metric, args = args
    dx, dv = _vector_field(t, (x, v), args)
    dlength = jnp.sqrt(jnp.dot(v, metric(x).mv(v)))
    return (dx, dv), dlength


@eqx.filter_jit
def solve_inference_ivp(
    v0: ArrayM,
    metric: Callable[[ArrayM], lx.AbstractLinearOperator],
    christoffel: Callable[[ArrayM], ArrayMMM],
    diffeq_solver: dfx.AbstractSolver = dfx.Tsit5(),
    diffeq_max_steps: None | int = 4096,
) -> tuple[Float[Array, "100 M"], FloatScalar]:
    term = dfx.ODETerm(_vector_field_with_length)
    t0 = 0
    t1 = 1
    dt0 = None
    y0 = (jnp.zeros_like(v0), v0)
    stepsize_controller = dfx.PIDController(
        rtol=1e-6, atol=1e-6, pcoeff=0.3, icoeff=0.4
    )
    trajectory_saveat = dfx.SubSaveAt(
        ts=jnp.linspace(t0, t1, 100), fn=lambda t, y, args: y[0][0]
    )
    length_saveat = dfx.SubSaveAt(t1=True, fn=lambda t, y, args: y[1])
    saveat = dfx.SaveAt(subs=(trajectory_saveat, length_saveat))
    sol = dfx.diffeqsolve(
        term,
        diffeq_solver,
        t0,
        t1,
        dt0,
        (y0, 0),
        args=(metric, christoffel),
        stepsize_controller=stepsize_controller,
        saveat=saveat,
        max_steps=diffeq_max_steps,
    )
    xs, (length,) = sol.ys
    return xs, length


def _solve_init(
    init_init_v0: ArrayM,
    init_x0: ArrayM,
    tdiff: ArrayM,
    christoffel: Callable[[ArrayM], ArrayMMM],
    solver: Union[optx.AbstractLeastSquaresSolver, optx.AbstractRootFinder, optx.AbstractMinimiser],
    max_steps: None | int,
) -> ArrayM:
    def _init_root_fn(init_v0: ArrayM, _) -> ArrayM:
        Γ = christoffel(init_x0)
        return (
            init_v0
            + 0.5 * tdiff * jnp.einsum("IJK,J,K->I", Γ, init_v0, init_v0)
            - init_init_v0
        )

    init_sol = optx.least_squares(_init_root_fn, solver, init_init_v0, max_steps=max_steps)
    return init_sol.value


# Also a good default:
# opt_solver = optx.OptaxMinimiser(optax.adabelief, rtol=1e-4, atol=1e-4, learning_rate=1e-2)
@eqx.filter_jit
def solve_bvp(
    target: ArrayM,
    pieces: int,
    christoffel: Callable[[ArrayM], ArrayMMM],
    canonicalise: Callable[[ArrayM], ArrayM] = lambda x: x,
    init_solver: Union[optx.AbstractLeastSquaresSolver, optx.AbstractRootFinder, optx.AbstractMinimiser] = optx.Newton(rtol=1e-4, atol=1e-4),
    diffeq_solver: dfx.AbstractSolver = dfx.Tsit5(scan_kind="lax"),  # TODO
    opt_solver: Union[optx.AbstractLeastSquaresSolver, optx.AbstractRootFinder, optx.AbstractMinimiser] = optx.Newton(rtol=1e-4, atol=1e-4),
    init_max_steps: None | int = 256,
    diffeq_max_steps: None | int = 4096,
    opt_max_steps: None | int = 256,
) -> ArrayM:
    ts = jnp.linspace(0, 1, pieces + 1)
    t0s = ts[:-1]
    t1s = ts[1:]
    (M,) = target.shape
    init_x0s: Float[Array, "pieces+1 M"] = jax.vmap(
        lambda x: jnp.linspace(0, x, pieces + 1, dtype=target.dtype), out_axes=1
    )(target)
    init_later_x0s: Float[Array, "pieces-1 M"] = init_x0s[1:-1]
    # For our multiple-shooting solve, we need to initialise both positions and
    # velocity at each of our boundary points.
    # We initialise positions as above: linearly interpolate in the space of intrinsic
    # coordinates.
    # Now to initialise velocities. We want these to approximately solve the shooting
    # problem over each interval, so that we take as few iterative steps as possible.
    #
    # Considering a single shooting interval, make a one-step Euler approximation over
    # the whole interval (here, `i` denote indexing over channels, and we elide the
    # indexing over intervals):
    #
    # x1_i = x0_i + \int_t0^t1 v_i(t) dt
    #      = x0_i + \int_t0^t1 v_i(t0) + \int_t0^t Γ^i_jk(s) v_j(s) v_k(s) ds dt
    #      ~ x0_i + \int_t0^t1 v_i(t0) + \int_t0^t Γ^i_jk(t0) v_j(t0) v_k(t0) ds dt
    #      = x0_i + \int_t0^t1 v_i(t0) + (t - t0) Γ^i_jk(t0) v_j(t0) v_k(t0) dt
    #      = x0_i + (t1 - t0) v_i(t0) + 0.5 (t1 - t0)^2 Γ^i_jk(t0) v_j(t0) v_k(t0)
    #                                                                                [A]
    # We see that an estimate of v(t0) is given by solving this multivariate
    # quadratric system. (Of the form 0 = A_ijk y_j y_k + B_ij y_j + c_i.)
    #
    # Now this is a difficult in general (unlike the linear case! i.e. A=0). See:
    #   https://mathoverflow.net/questions/153436/can-you-efficiently-solve-a-system-of-quadratic-multivariate-polynomials
    #   https://arxiv.org/abs/cs/0403008
    #   https://hal.science/hal-01567408/document
    #
    # So, the answer is to solve this in turn as a root-finding problem.
    #
    # Now we note that our Γ is often sparse/small. So to solve this root-finding
    # problem in turn, we start off our iteration with v(t0) = (x1 - x0)/(t1 - t0).
    # (Recall that we are still considering just a single shooting interval.). Then
    # run a root-finding algorithm wrt [A] over every interal.
    #
    # Then returning to our overall problem: use *that* to initialise our multiple
    # shooting algorithm!
    tdiffs: Float[Array, "pieces 1"] = (t1s - t0s)[:, jnp.newaxis]
    init_init_v0s: Float[Array, "pieces M"] = (init_x0s[1:] - init_x0s[:-1]) / tdiffs
    # For efficiency, solve this as a batch-of-root-finding-problems, not just a single
    # big root-finding problem. (Scales as O(BM^3) rather than O((BM)^3).)
    init_v0s = eqx.filter_vmap(_solve_init)(
        init_init_v0s, init_x0s[:-1], tdiffs, christoffel, init_solver, init_max_steps
    )

    target = canonicalise(target)

    def _root_fn(
        later_x0s: Float[Array, "pieces-1 M"], v0s: Float[Array, "pieces M"]
    ):  # M * (2 * pieces - 1) inputs
        zero = jnp.zeros((1, M), target.dtype)
        x0s = jnp.concatenate([zero, later_x0s])
        (x1s, v1s) = eqx.filter_vmap(_solve_single_ivp)(t0s, t1s, x0s, v0s, christoffel, diffeq_solver, diffeq_max_steps)
        x0s = jax.vmap(canonicalise)(x0s)
        x1s = jax.vmap(canonicalise)(x1s)
        x_diff = x0s[1:] - x1s[:-1]  # M * (pieces - 1) conditions
        v_diff = v0s[1:] - v1s[:-1]  # M * (pieces - 1) conditions
        target_diff = x1s[-1] - target  # M conditions
        return x_diff, v_diff, target_diff  # M * (2 * pieces - 1) outputs

    def _lstsq_fn(z, _):
        return _root_fn(*z)

    sol = optx.least_squares(
        _lstsq_fn, opt_solver, (init_later_x0s, init_v0s), max_steps=opt_max_steps, throw=False
    )
    _, v0s = sol.value
    return v0s[0]


# Source: https://www.geometrictools.com/Documentation/RiemannianGeodesics.pdf
def manifold_to_metric(
    manifold: Callable[[ArrayM], Array1D]
) -> Callable[[ArrayM], lx.AbstractLinearOperator]:
    def manifold_with_args(x, args):
        del args
        return manifold(x)

    @jax.jit
    def metric(x: ArrayM) -> ArrayMM:
        jac = lx.JacobianLinearOperator(manifold_with_args, x)
        jac = lx.linearise(jac)
        return jac.T @ jac

    return metric


# Source: https://www.geometrictools.com/Documentation/RiemannianGeodesics.pdf
def metric_to_christoffel(
    metric: Callable[[ArrayM], lx.AbstractLinearOperator]
) -> Callable[[ArrayM], ArrayMMM]:
    def materialised_metric(x):
        return metric(x).as_matrix()

    def christoffel_first_kind(x: ArrayM) -> ArrayMMM:
        jac_ijk: ArrayMMM = jax.jacfwd(materialised_metric)(x)
        jac_jki = jnp.transpose(jac_ijk, (1, 2, 0))
        jac_kij = jnp.transpose(jac_ijk, (2, 0, 1))
        return 0.5 * (jac_jki + jac_kij - jac_ijk)

    @jax.jit
    def christoffel_second_kind(x: ArrayM) -> ArrayMMM:
        Γ: ArrayMMM = christoffel_first_kind(x)
        g = metric(x)
        if lx.is_diagonal(g):
            diag: ArrayM = lx.diagonal(g)
            g_inv: ArrayM = 1 / diag
            return Γ * g_inv
        else:
            g_inv: ArrayMM = jnp.linalg.inv(g.as_matrix())
            return jnp.tensordot(g_inv, Γ, axes=((1,), (2,)))  # pyright: ignore

    return christoffel_second_kind


def demo_qubit():
    N = 1  # number of qubits
    M = 4**N - 1  # number of real values parameterising SU(2^N)

    # Basis elements: (for N>1, tensor products of) Pauli matrices
    σ0: Complex[ndarray, "2*N 2*N"] = np.array([[0, 1], [1, 0.0 + 0j]])
    σ1: Complex[ndarray, "2*N 2*N"] = np.array([[0, -1j], [1j, 0]])  # pyright: ignore
    σ2: Complex[ndarray, "2*N 2*N"] = np.array([[1, 0], [0, -1.0 + 0j]])
    σ_vec: Complex[ndarray, "M 2*N 2*N"] = np.stack([σ0, σ1, σ2])
    assert σ_vec.shape == (M, 2 * N, 2 * N)

    # Products of basis elements
    bbmm = jax.vmap(jax.vmap(operator.matmul, in_axes=(None, 0)), in_axes=(0, None))
    σ_prod: Complex[Array, "M M 2*N 2*N"] = bbmm(σ_vec, σ_vec)  # σ_prod[i,j]=σ_i@σ_j
    assert σ_prod.shape == (M, M, 2 * N, 2 * N)

    # Moments of inertia
    I_diag: Float[Array, " M"] = jnp.array([0.1, 0.1, 1.0])
    assert I_diag.shape == (M,)

    # Trace[A^H B], normalised s.t. Trace[Id] = 1
    def inner_product(A: Complex[Array, "2*N 2*N"], B: Complex[Array, "2*N 2*N"]) -> Complex[Array, ""]:
        return jnp.sum(A.conj().T * B) / (2 * N)

    # The metric g_ij(x) for fixed i,j,x.
    def metric_element(
        σ_prod1: Complex[Array, "M 2*N 2*N"],  # σ_k @ σ_i with varying k and fixed i
        σ_prod2: Complex[Array, "M 2*N 2*N"],  # σ_k @ σ_j with varying k and fixed j
        x: Float[Array, " M"],
    ) -> Float[Array, ""]:
        U: Complex[Array, "2*N 2*N"] = jsp.linalg.expm(jnp.einsum("a,abc->bc", 1j * x, σ_vec))
        batch_inner_product = jax.vmap(inner_product, in_axes=(None, 0))
        left: Complex[Array, "M"] = batch_inner_product(U, σ_prod1)
        right: Complex[Array, "M"] = batch_inner_product(U, σ_prod2)
        return jnp.einsum("i,i,i->", I_diag, left, right).real

    # The metric at a particular set of intrinsic coordinates x.
    def metric(x: Float[Array, " M"]) -> lx.AbstractLinearOperator:
        bb_metric_element = jax.vmap(
            jax.vmap(metric_element, in_axes=(None, 1, None)), in_axes=(1, None, None)
        )
        matrix: Float[Array, "M M"] = bb_metric_element(σ_prod, σ_prod, x)
        return lx.MatrixLinearOperator(matrix)

    christoffel = metric_to_christoffel(metric)

    def canonicalise(x: Float[Array, " M"]):
        return x % (2 * np.pi)

    for t in np.linspace(1, 6, 10):
        target = np.array([0.4387, 0.3816, 0]) * t
        pieces = 40
        init_solver = optx.OptaxMinimiser(optax.adabelief, rtol=1e-4, atol=1e-4, learning_rate=1e-2)
        diffeq_solver = dfx.Tsit5()
        opt_solver = optx.OptaxMinimiser(optax.adabelief, rtol=1e-4, atol=1e-4, learning_rate=1e-3)
        v0 = solve_bvp(target, pieces, christoffel, canonicalise, init_solver=init_solver, diffeq_solver=diffeq_solver, opt_solver=opt_solver)
        xs, length = solve_inference_ivp(v0, metric, christoffel, diffeq_max_steps=16**5)
        print(f"{t=:.3f}, length={length.item():.3f}")


# Source: https://www.cefns.nau.edu/~schulz/torus.pdf
def demo_torus():
    R = 1
    r = 0.2

    def manifold(x: Float[Array, "2"]) -> Float[Array, "3"]:
        assert x.shape == (2,)
        θ, φ = x
        tmp = R + r * jnp.cos(φ)
        x = tmp * jnp.cos(θ)
        y = tmp * jnp.sin(θ)
        z = r * jnp.sin(φ)
        return jnp.stack([x, y, z])

    # Or `metric = manifold_to_metric(manifold)`, but this is more efficient.
    def metric(x: Float[Array, "2"]) -> lx.AbstractLinearOperator:
        assert x.shape == (2,)
        _, φ = x
        g_θθ = (R + r * jnp.cos(φ)) ** 2
        g_φφ = r**2
        return lx.DiagonalLinearOperator(jnp.array([g_θθ, g_φφ]))

    # Or `christoffel = metric_to_christoffel(metric)`, but this is more efficient.
    def christoffel(x: Float[Array, "2"]) -> Float[Array, "2 2 2"]:
        assert x.shape == (2,)
        θ, φ = x
        tmp = R + r * jnp.cos(φ)
        Γ⁀θ_θθ = 0
        Γ⁀θ_θφ = Γ⁀θ_φθ = -r * jnp.sin(φ) / tmp
        Γ⁀θ_φφ = 0
        Γ⁀φ_θθ = tmp * jnp.sin(φ) / r
        Γ⁀φ_θφ = 0
        Γ⁀φ_φθ = 0
        Γ⁀φ_φφ = 0
        return jnp.array(
            [[[Γ⁀θ_θθ, Γ⁀θ_θφ], [Γ⁀θ_φθ, Γ⁀θ_φφ]], [[Γ⁀φ_θθ, Γ⁀φ_θφ], [Γ⁀φ_φθ, Γ⁀φ_φφ]]]
        )

    def canonicalise(x: Float[Array, "2"]) -> Float[Array, "2"]:
        assert x.shape == (2,)
        return x % (2 * jnp.pi)

    θ_target = 1
    φ_target = 2.2
    target = jnp.array([θ_target, φ_target])
    pieces = 4  # TODO: increasing this doesn't seem to help?
    v0 = solve_bvp(target, pieces, christoffel, canonicalise)
    xs, length = solve_inference_ivp(v0, metric, christoffel)
    print(f"length={length.item():.3f}")
    x_path, y_path, z_path = jax.vmap(manifold, out_axes=1)(xs)
    x0, y0, z0 = manifold(jnp.zeros_like(target))
    x_target, y_target, z_target = manifold(target)

    surface = np.mgrid[0.0 : 2 * np.pi : 100j, 0.0 : 2 * np.pi : 100j]
    x_surface, y_surface, z_surface = jax.vmap(
        jax.vmap(manifold, in_axes=1, out_axes=1), in_axes=1, out_axes=1
    )(surface)  # pyright: ignore
    fig = plt.figure()
    ax = fig.add_subplot(121, projection="3d")
    ax.plot_surface(
        x_surface, y_surface, z_surface, rstride=1, cstride=1, color="c", linewidth=0
    )
    ax.plot(
        1.02 * x_path, 1.02 * y_path, 1.02 * z_path, color="red", linewidth=3, zorder=10
    )
    ax.plot(1.02 * x0, 1.02 * y0, 1.02 * z0, "o", zorder=20, c="blue")
    ax.plot(
        1.02 * x_target, 1.02 * y_target, 1.02 * z_target, "o", zorder=20, c="orange"
    )
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.set_aspect("equal")
    ax = fig.add_subplot(122)
    ax.plot(xs[:, 0], xs[:, 1], color="red")
    ax.plot(0, 0, "o", c="blue")
    ax.plot(*target, "o", c="orange")
    ax.set_xlim([0, 2 * np.pi])
    ax.set_ylim([0, 2 * np.pi])
    ax.set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    # demo_torus()
    demo_qubit()


# TODO:
# - Failing to integrate the final single shooting piece -- why?
# - Te initial z-component is flip-flopping back and forth at init. Why?
