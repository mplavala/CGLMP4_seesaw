import numpy as np
import picos as pc

ATOL = 1E-5


def is_herm(matrix):
    """Check whether `matrix` is Hermitian."""

    return np.allclose(matrix, matrix.conj().T)


def is_psd(matrix):
    """Test `matrix` for positive semi-definiteness."""

    if is_herm(matrix):
        return np.all(np.linalg.eigvalsh(matrix) >= - ATOL)
    else:
        return False


def is_measurement(meas: list[np.ndarray], verbose :bool = False) -> bool:
    """Check whether `meas` is a well-defined quantum measurement.

    Args:
        meas (list): list of ndarrays representing the measurement's effects.
        verbose: if true prints which criterion failed.

    Returns:
        bool: returns `True` iff `meas` is composed of effects which are:
            - Square matrices with the same dimension,
            - Positive semi-definite, and
            - Sum to the identity operator.
    """

    dims = meas[0].shape

    try:
        square = len(dims) == 2 and dims[0] == dims[1]
        same_dim = np.all([effect.shape == dims for effect in meas])
        psd = np.all([is_psd(effect) for effect in meas])
        complete = np.allclose(sum(meas), np.eye(dims[0]))
    except (ValueError, np.linalg.LinAlgError):
        return False

    if verbose:
        if not square: print("Not square matrixes.")
        if not same_dim: print("Matrixes don't have the same dimensions.")
        if not psd: print("Some matrixes are not PSD.")
        if not complete: print("POVM does not sum to identity.")

    return square and same_dim and psd and complete


def is_assemblage(assemblage: list[list[np.ndarray]], verbose :bool = False) -> bool:
    """Check whether `assemblage` is a well-defined quantum measurement.

    Args:
        assemblage (list): list of lists of ndarrays representing the assemblage.
        verbose: if true prints which criterion failed.

    Returns:
        bool: returns `True` iff `assemblage` is composed of effects which are:
            - Square matrices with the same dimension,
            - Positive semi-definite,
            - Non-signaling, and
            - Normalized
    """
    dims = assemblage[0][0].shape
    nx = len(assemblage)
    state = np.sum(assemblage[0], axis=0)

    try:
        square = len(dims) == 2 and dims[0] == dims[1]
        same_dim = np.all([[sigma.shape == dims for sigma in assemblage[x]] for x in range(nx)])
        psd = np.all([[is_psd(sigma) for sigma in assemblage[x]] for x in range(nx)])
        non_signaling = np.all([np.allclose(sum(assemblage[x]), state) for x in range(nx)])
        normalized = np.allclose(np.trace(state), 1)
    except (ValueError, np.linalg.LinAlgError):
        return False

    if verbose:
        if not square: print("Not square matrixes.")
        if not same_dim: print("Matrixes don't have the same dimensions.")
        if not psd: print("Some matrixes are not PSD.")
        if not non_signaling: print("Assemblage is signaling.")
        if not normalized: print("Assemlbage is not normalized.")

    return square and same_dim and psd and non_signaling and normalized


def get_random_PVM(dim: int = 2):
    assert dim >= 2

    random = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
    hermitian = random @ np.transpose(np.conjugate(random))

    vectors = np.linalg.eigh(hermitian)[1]

    measurement = [np.outer(vector.conj(), vector) for vector in vectors]

    assert is_measurement(measurement, verbose=True)

    return measurement


def CGLMP_4(measurements: list[list[np.ndarray]], assemblage: list[list[np.ndarray]]):
    """Computes the CGLMP_4 value for list of measurements and an assemblage."""
    
    assert len(measurements) == len(assemblage)
    for meas in measurements:
        assert is_measurement(meas)
    assert is_assemblage(assemblage)

    return sum([np.trace(assemblage[0][b] @ measurements[0][a]) for a in range(4) for b in range(4) if a <= b]) + sum([np.trace(assemblage[1][b] @ measurements[0][a]) for a in range(4) for b in range(4) if a >= b]) + sum([np.trace(assemblage[0][b] @ measurements[1][a]) for a in range(4) for b in range(4) if a >= b]) - sum([np.trace(assemblage[1][b] @ measurements[1][a]) for a in range(4) for b in range(4) if a >= b]) - 2


def get_preparation_SDP(measurements: list[list[np.ndarray]]):

    SDP = pc.Problem()

    # Constants
    meas_xa = [[pc.Constant(f"M_{a}|{x}", measurements[x][a]) for a in range(4)] for x in range(2)]

    # Variables
    ass_yb = [[pc.HermitianVariable(f"sigma_{b}|{y}", 4) for b in range(4)] for y in range(2)]
    val = pc.RealVariable("val")

    # Assemblage constraints
    SDP.add_list_of_constraints([ass_yb[y][b] >> 0 for b in range(4) for y in range(2)])
    SDP.add_constraint(pc.sum(ass_yb[0]) == pc.sum(ass_yb[1]))
    SDP.add_constraint(pc.trace(pc.sum(ass_yb[0])) == 1)

    # Separable constraints
    SDP.add_list_of_constraints([ass_yb[y][b].partial_transpose(0, [2, 2]) >> 0 for b in range(4) for y in range(2)])

    # Maximazied variable
    SDP.add_constraint(val == sum([pc.trace(ass_yb[0][b] * meas_xa[0][a]) for a in range(4) for b in range(4) if a <= b]) +
                              sum([pc.trace(ass_yb[1][b] * meas_xa[0][a]) for a in range(4) for b in range(4) if a >= b]) +
                              sum([pc.trace(ass_yb[0][b] * meas_xa[1][a]) for a in range(4) for b in range(4) if a >= b]) -
                              sum([pc.trace(ass_yb[1][b] * meas_xa[1][a]) for a in range(4) for b in range(4) if a >= b]) - 2)

    SDP.set_objective("max", val)

    return SDP


def get_measurement_SDP(assemblage: list[list[np.ndarray]]):

    SDP = pc.Problem()

    # Constants
    ass_yb = [[pc.Constant(f"sigma_{b}|{y}", assemblage[y][b]) for b in range(4)] for y in range(2)]
    I = pc.Constant("I", np.eye(4))

    # Variables
    meas_xa = [[pc.HermitianVariable(f"M_{a}|{x}", 4) for a in range(4)] for x in range(2)]
    val = pc.RealVariable("val")

    # POVMs constraints
    SDP.add_list_of_constraints([meas_xa[x][a] >> 0 for a in range(4) for x in range(2)])
    SDP.add_constraint(pc.sum(meas_xa[0]) == I)
    SDP.add_constraint(pc.sum(meas_xa[1]) == I)

    # Maximazied variable
    SDP.add_constraint(val == sum([pc.trace(ass_yb[0][b] * meas_xa[0][a]) for a in range(4) for b in range(4) if a <= b]) +
                              sum([pc.trace(ass_yb[1][b] * meas_xa[0][a]) for a in range(4) for b in range(4) if a >= b]) +
                              sum([pc.trace(ass_yb[0][b] * meas_xa[1][a]) for a in range(4) for b in range(4) if a >= b]) -
                              sum([pc.trace(ass_yb[1][b] * meas_xa[1][a]) for a in range(4) for b in range(4) if a >= b]) - 2)

    SDP.set_objective("max", val)

    return SDP


def see_saw(measurements: list[list[np.ndarray]]):
    res = [-1] * 10
    I_4 = 1

    while not np.all([np.allclose(prev_res, I_4, atol=1E-4) for prev_res in res]):
    
        # Assemblage optimization
        assert is_measurement(measurements[0], verbose=True)
        assert is_measurement(measurements[1], verbose=True)
        ass_SDP = get_preparation_SDP(measurements)
        ass_SDP.solve(solver="mosek", verbosity=0)
        assemblage = [[ass_SDP.get_variable(f"sigma_{b}|{y}").np2d for b in range(4)] for y in range(2)]
    
        # Measurement optimization
        assert is_assemblage(assemblage, verbose=True)
        meas_SDP = get_measurement_SDP(assemblage)
        meas_SDP.solve(solver="mosek", verbosity=0)
        measurements = [[meas_SDP.get_variable(f"M_{a}|{x}").np2d for a in range(4)] for x in range(2)]
    
        I_4 = np.real_if_close(CGLMP_4(measurements, assemblage))
    
        res.pop(0)
        res.append(I_4)

    return [I_4, measurements, assemblage]


if __name__ == '__main__':
    N = 10
    results = []

    for n in range(1, N+1):
        print(f"n = {n}")
        initial_measurements = [get_random_PVM(4), get_random_PVM(4)]
        [I_4, measurements, assemblage] = see_saw(initial_measurements)

        print(f"I_4 = {I_4}")
        results.append(I_4)

    print()
    print(f"Best result = {np.max(results)}")
    
    np.save(f"results-N-{N}-I_4-{np.max(results)}", results)

