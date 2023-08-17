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
from jaxtyping import Array, ArrayLike, Complex, Float, Int, PyTree
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from numpy import ndarray
from typing import Optional, TypeAlias, Union


jax.config.update("jax_enable_x64", True)


# M = 4^N - 1 for a system of N qubits.
FloatScalar: TypeAlias = Float[Array, ""]
IntScalar: TypeAlias = Int[Array, ""]
Array1D: TypeAlias = Float[Array, " _"]
ArrayM: TypeAlias = Float[Array, " M"]
ArrayMM: TypeAlias = Float[Array, "M M"]
ArrayMMM: TypeAlias = Float[Array, "M M M"]


def _vector_field(
    t: FloatScalar, y: tuple[ArrayM, ArrayM], christoffel: Callable[[ArrayM], ArrayMMM]
) -> tuple[ArrayM, ArrayM]:
    x, v = y
    dx = v
    dv = -jnp.einsum("IJK,J,K->I", christoffel(x), v, v)
    return dx, dv


def _solve_single_ivp(
    t0: FloatScalar,
    t1: FloatScalar,
    x0: ArrayM,
    v0: ArrayM,
    christoffel: Callable[[ArrayM], ArrayMMM],
    solver: dfx.AbstractSolver,
    adjoint: dfx.AbstractAdjoint,
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
        solver,
        t0,
        t1,
        dt0,
        y0,
        args=christoffel,
        stepsize_controller=stepsize_controller,
        adjoint=adjoint,
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
    saveat: dfx.SaveAt,
    t1: Float[ArrayLike, ""] = 1,
    solver: dfx.AbstractSolver = dfx.Tsit5(),
    max_steps: None | int = 4096,
) -> tuple[PyTree[Array], IntScalar]:
    term = dfx.ODETerm(_vector_field_with_length)
    t0 = 0
    dt0 = None
    y0 = (jnp.zeros_like(v0), v0)
    stepsize_controller = dfx.PIDController(
        rtol=1e-10, atol=1e-10, pcoeff=0.3, icoeff=0.4
    )
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        (y0, 0),
        args=(metric, christoffel),
        stepsize_controller=stepsize_controller,
        saveat=saveat,
        max_steps=max_steps,
    )
    return sol.ys, sol.stats["num_steps"]

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

    init_sol = optx.root_find(_init_root_fn, solver, init_init_v0, max_steps=max_steps)
    return init_sol.value


# Also a good default:
# opt_solver = optx.OptaxMinimiser(optax.adabelief, rtol=1e-4, atol=1e-4, learning_rate=1e-2)
@eqx.filter_jit
def solve_bvp(
    target: ArrayM,
    pieces: int,
    christoffel: Callable[[ArrayM], ArrayMMM],
    bounds: Optional[ArrayM] = None,
    init_init_v0s: Optional[Float[Array, "pieces M"]] = None,  # pyright: ignore
    init_solver: Union[None, optx.AbstractLeastSquaresSolver, optx.AbstractRootFinder, optx.AbstractMinimiser] = optx.Newton(rtol=1e-4, atol=1e-4),
    diffeq_solver: dfx.AbstractSolver = dfx.Tsit5(),
    opt_solver: Union[optx.AbstractLeastSquaresSolver, optx.AbstractRootFinder, optx.AbstractMinimiser] = optx.Newton(rtol=1e-4, atol=1e-4),
    diffeq_adjoint: dfx.AbstractAdjoint = dfx.RecursiveCheckpointAdjoint(),
    init_max_steps: None | int = 256,
    diffeq_max_steps: None | int = 4096,
    opt_max_steps: None | int = 256,
) -> tuple[ArrayM, IntScalar]:
    if bounds is not None:
        # We always start our diffeq at zero, so put the cut on the opposite side.
        offset = jnp.where(jnp.isinf(bounds), 0, (bounds / 2))
        target = ((target + offset) % bounds) - offset
    ts = jnp.linspace(0, 1, pieces + 1)
    t0s = ts[:-1]
    t1s = ts[1:]
    (M,) = target.shape
    init_x0s: Float[Array, "pieces+1 M"] = jax.vmap(
        lambda x: jnp.linspace(0, x, pieces + 1, dtype=target.dtype), out_axes=1
    )(target)
    init_later_x0s: Float[Array, "pieces-1 M"] = init_x0s[1:-1]
    if init_solver is None:
        if init_init_v0s is None:
            init_init_v0s: Float[Array, "pieces M"]  = jnp.zeros((pieces, M), target.dtype)
        init_v0s: Float[Array, "pieces M"] = init_init_v0s
    else:
        # For our multiple-shooting solve, we need to initialise both positions and
        # velocity at each of our boundary points.
        # We initialise positions as above: linearly interpolate in the space of
        # intrinsic coordinates.
        # Now to initialise velocities. We want these to approximately solve the
        # shooting problem over each interval, so that we take as few iterative steps as
        # possible.
        #
        # Considering a single shooting interval, make a one-step Euler approximation
        # over the whole interval (here, `i` denote indexing over channels, and we elide
        # the indexing over intervals):
        #
        # x1_i = x0_i + \int_t0^t1 v_i(t) dt
        #      = x0_i + \int_t0^t1 v_i(t0) + \int_t0^t Γ^i_jk(s) v_j(s) v_k(s) ds dt
        #      ~ x0_i + \int_t0^t1 v_i(t0) + \int_t0^t Γ^i_jk(t0) v_j(t0) v_k(t0) ds dt
        #      = x0_i + \int_t0^t1 v_i(t0) + (t - t0) Γ^i_jk(t0) v_j(t0) v_k(t0) dt
        #      = x0_i + (t1 - t0) v_i(t0) + 0.5 (t1 - t0)^2 Γ^i_jk(t0) v_j(t0) v_k(t0)
        #                                                                            [A]
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
        if init_init_v0s is None:
            init_init_v0s: Float[Array, "pieces M"] = (init_x0s[1:] - init_x0s[:-1]) / tdiffs
        # For efficiency, solve this as a batch-of-root-finding-problems, not just a single
        # big root-finding problem. (Scales as O(BM^3) rather than O((BM)^3).)
        init_v0s = eqx.filter_vmap(_solve_init)(
            init_init_v0s, init_x0s[:-1], tdiffs, christoffel, init_solver, init_max_steps
        )

    def _root_fn(inputs, _):
        later_x0s, v0s = inputs  # M * (2 * pieces - 1) inputs
        later_x0s: Float[Array, "pieces-1 M"]
        v0s: Float[Array, "pieces M"]
        zero = jnp.zeros((1, M), target.dtype)
        x0s = jnp.concatenate([zero, later_x0s])
        (x1s, v1s) = eqx.filter_vmap(_solve_single_ivp)(t0s, t1s, x0s, v0s, christoffel, diffeq_solver, diffeq_adjoint, diffeq_max_steps)
        # TODO: figure out bounds
        x_diff = x0s[1:] - x1s[:-1]  # M * (pieces - 1) conditions
        v_diff = v0s[1:] - v1s[:-1]  # M * (pieces - 1) conditions
        target_diff = x1s[-1] - target  # M conditions
        return x_diff, v_diff, target_diff  # M * (2 * pieces - 1) outputs

    sol = optx.root_find(
        _root_fn, opt_solver, (init_later_x0s, init_v0s), max_steps=opt_max_steps
    )
    _, v0s = sol.value
    return v0s[0], sol.stats["num_steps"]


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
        return jnp.sum(A.conj() * B) / (2 * N)

    # The metric at a particular set of intrinsic coordinates x.
    def metric(x: Float[Array, " M"]) -> lx.AbstractLinearOperator:
        U: Complex[Array, "2*N 2*N"] = jsp.linalg.expm(jnp.einsum("a,abc->bc", 1j * x, σ_vec))
        bb_inner_product = jax.vmap(jax.vmap(inner_product, in_axes=(None, 0)), in_axes=(None, 0))
        traces: Complex[Array, "M M"] = bb_inner_product(U, σ_prod)
        matrix: Complex[Array, "M M"] = jnp.einsum("i,ik,im->km", I_diag, traces, traces)
        return lx.MatrixLinearOperator(matrix.real)

    # For debugging purposes (with N=1).
    # def metric_manual(x: Float[Array, " M"]) -> lx.AbstractLinearOperator:
    #     U = jsp.linalg.expm(1j * (x[0] * σ0 + x[1] * σ1 + x[2] * σ2))
    #     a00 = jnp.trace(-σ0 @ U.conj().T @ σ0) / 2
    #     a01 = jnp.trace(-σ1 @ U.conj().T @ σ0) / 2
    #     a02 = jnp.trace(-σ2 @ U.conj().T @ σ0) / 2
    #     a10 = jnp.trace(-σ0 @ U.conj().T @ σ1) / 2
    #     a11 = jnp.trace(-σ1 @ U.conj().T @ σ1) / 2
    #     a12 = jnp.trace(-σ2 @ U.conj().T @ σ1) / 2
    #     a20 = jnp.trace(-σ0 @ U.conj().T @ σ2) / 2
    #     a21 = jnp.trace(-σ1 @ U.conj().T @ σ2) / 2
    #     a22 = jnp.trace(-σ2 @ U.conj().T @ σ2) / 2
    #     I0, I1, I2 = I_diag
    #     g00 = I0 * a00*a00 + I1 * a10*a10 + I2 * a20*a20
    #     g01 = I0 * a00*a01 + I1 * a10*a11 + I2 * a20*a21
    #     g02 = I0 * a00*a02 + I1 * a10*a12 + I2 * a20*a22
    #     g10 = I0 * a01*a00 + I1 * a11*a10 + I2 * a21*a20
    #     g11 = I0 * a01*a01 + I1 * a11*a11 + I2 * a21*a21
    #     g12 = I0 * a01*a02 + I1 * a11*a12 + I2 * a21*a22
    #     g20 = I0 * a02*a00 + I1 * a12*a10 + I2 * a22*a20
    #     g21 = I0 * a02*a01 + I1 * a12*a11 + I2 * a22*a21
    #     g22 = I0 * a02*a02 + I1 * a12*a12 + I2 * a22*a22
    #     matrix = jnp.array([[g00, g01, g02], [g10, g11, g12], [g20, g21, g22]]).real
    #     return lx.MatrixLinearOperator(matrix)

    christoffel = metric_to_christoffel(metric)

    bounds = jnp.array([2 * np.pi, 2 * np.pi, 2 * np.pi])
    pieces = 4
    init_v0s = jnp.zeros((pieces, M))  # Used on the i=0 and i=1 iterations.
    init_solver = None
    optim = optax.chain(
        optax.adam(1e-3),
        optax.scale_by_schedule(
            optax.piecewise_constant_schedule(1, {200: 0.1})
        )
    )
    opt_solver = optx.OptaxMinimiser(lambda: optim, rtol=1e-4, atol=1e-4)

    ts = np.linspace(0, 2, 21)

    for i in range(len(ts)):
        t = ts[i]
        next_t = ts[min(i + 1, len(ts) - 1)]
        target = jnp.array([0.4387, 0.3816, 0]) * t
        v0, optim_num_steps = solve_bvp(target, pieces, christoffel, bounds, init_init_v0s=init_v0s, init_solver=init_solver, opt_solver=opt_solver, opt_max_steps=5000)

        if t == 0:
            next_t1 = jnp.array(1.0)
        else:
            next_t1 = next_t / t
        x1_saveat = dfx.SubSaveAt(ts=jnp.array([1.0]), fn=lambda t, y, args: y[0][0])
        vs_saveat = dfx.SubSaveAt(ts=jnp.linspace(0, next_t1, pieces + 1)[:-1], fn=lambda t, y, args: y[0][1])
        length_saveat = dfx.SubSaveAt(ts=jnp.array([1.0]), fn=lambda t, y, args: y[1])
        saveat = dfx.SaveAt(subs=(x1_saveat, vs_saveat, length_saveat))
        ((x1,), _init_v0s, (length,)), diffeq_num_steps = solve_inference_ivp(v0, metric, christoffel, t1=next_t1, saveat=saveat, max_steps=16**5)
        if t != 0:
            init_v0s = _init_v0s
            
        error = (x1 - target).tolist()
        error_string = "[" + ", ".join(f"{x:.5f}" for x in error) + "]"
        print(f"{t=:.3f}, length={length.item():.3f} error={error_string} optim_num_steps={optim_num_steps.item()} diffeq_num_steps={diffeq_num_steps.item()}")


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

    bounds = jnp.array([2 * np.pi, 2 * np.pi])

    θ_target = 1
    φ_target = 2.2
    target = jnp.array([θ_target, φ_target])
    pieces = 4
    v0, _ = solve_bvp(target, pieces, christoffel, bounds, diffeq_solver=dfx.Tsit5(scan_kind="lax"), diffeq_adjoint=dfx.DirectAdjoint())

    xs_saveat = dfx.SubSaveAt(ts=jnp.linspace(0, 1, 100), fn=lambda t, y, args: y[0][0])
    length_saveat = dfx.SubSaveAt(t1=True, fn=lambda t, y, args: y[1])
    saveat = dfx.SaveAt(subs=(xs_saveat, length_saveat))
    (xs, (length,)), _ = solve_inference_ivp(v0, metric, christoffel, saveat=saveat)
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
