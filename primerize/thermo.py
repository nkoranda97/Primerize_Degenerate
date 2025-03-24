import math
import numpy

numpy.seterr("ignore")


class Singleton(object):
    """Base class for singleton pattern"""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance


class Nearest_Neighbor_Parameter(Singleton):
    """Wrapper object of Nearest Neighbor parameters; for internal use.

    Attributes:
        T: ``float``
        delH_NN: ``numpy.array(float(4, 4))``
        delS_NN: ``numpy.array(float(4, 4))``
        delG_NN: ``numpy.array(float(4, 4))``
        delH_AT_closing_penalty: ``numpy.array(float(1, 4))``
        delS_AT_closing_penalty: ``numpy.array(float(1, 4))``
        delG_AT_closing_penalty: ``numpy.array(float(1, 4))``
        delH_mismatch: ``numpy.array(float(4, 4, 4))``
        delS_mismatch: ``numpy.array(float(4, 4, 4))``
        delG_mismatch: ``numpy.array(float(4, 4, 4))``
        delH_init: ``float``
        del_init: ``float``

    Note:
        This ``class`` follows the singleton pattern so that only one instance is created. An instance is already initialized as ``primerize.Nearest_Neighbor``.
    """

    def __init__(self):
        self.T = 273.15 + 37

        # From SantaLucia, Jr, Ann Rev 2004
        # I was lazy and used AI to add the N, K, M values.
        # For N I had it use the average of A, C, G, T.
        # For K it averaged G and T.
        # For M it averaged A and C.

        # A  C  G  T  N  K  M
        self.delH_NN = numpy.array(
            [
                [-7.6, -8.4, -7.8, -7.2, -7.75, -7.5, -8.0],  # A
                [-8.5, -8.0, -9.8, -7.8, -8.525, -8.8, -8.25],  # C
                [-8.2, -9.8, -8.0, -8.4, -8.6, -8.2, -9.0],  # G
                [-7.2, -8.2, -8.5, -7.6, -7.875, -8.05, -7.7],  # T
                [-7.875, -8.6, -8.525, -7.75, -8.1875, -8.1375, -8.2375],  # N
                [-7.7, -9.0, -8.25, -8.0, -8.2375, -8.125, -8.35],  # K
                [-8.05, -8.2, -8.8, -7.5, -8.1375, -8.15, -8.125],  # M
            ]
        )
        self.delS_NN = numpy.array(
            [
                [-21.3, -22.4, -21.0, -20.4, -21.275, -20.7, -21.85],  # A
                [-22.7, -19.9, -24.4, -21.0, -22.0, -22.7, -21.3],  # C
                [-22.2, -24.4, -19.9, -22.4, -22.225, -21.15, -23.3],  # G
                [-21.3, -22.2, -22.7, -21.3, -21.875, -22.0, -21.75],  # T
                [-21.875, -22.225, -22.0, -21.275, -21.84375, -21.6375, -22.05],  # N
                [-21.75, -23.3, -21.3, -21.85, -22.05, -21.575, -22.525],  # K
                [-22.0, -21.15, -22.7, -20.7, -21.6375, -21.7, -21.575],  # M
            ]
        )
        self.delG_NN = self.delH_NN - (self.T * self.delS_NN) / 1000

        # From SantaLucia, Jr, Ann Rev 2004
        # A  C  G  T  N  K  M
        self.delH_AT_closing_penalty = numpy.array([2.2, 0.0, 0.0, 2.2, 1.1, 1.1, 1.1])
        self.delS_AT_closing_penalty = numpy.array(
            [6.9, 0.0, 0.0, 6.9, 3.45, 3.45, 3.45]
        )
        self.delG_AT_closing_penalty = (
            self.delH_AT_closing_penalty
            - (self.T * self.delS_AT_closing_penalty) / 1000
        )

        # Following also from Santaucia/Hicks.
        self.delH_mismatch = numpy.zeros((7, 7, 7))
        # AX/TY
        self.delH_mismatch[:, :, 0] = [
            [1.2, 2.3, -0.6, -7.6, -1.175, -4.1, 1.75],  # A
            [5.3, 0.0, -8.4, 0.7, -0.6, -3.85, 2.65],  # C
            [-0.7, -7.8, -3.1, 1.0, -2.65, -1.05, -4.25],  # G
            [-7.2, -1.2, -2.5, -2.7, -3.4, -2.6, -4.2],  # T
            [-0.35, -1.675, -3.65, -2.15, -1.95, -2.9, -1.0125],  # N
            [-3.95, -4.5, -2.8, -0.85, -3.025, -1.825, -4.225],  # K
            [3.25, 1.15, -4.5, -3.45, -0.8875, -3.975, 2.2],  # M
        ]
        # CX/GY
        self.delH_mismatch[:, :, 1] = [
            [-0.9, 1.9, -0.7, -8.5, -2.05, -4.6, 0.5],  # A
            [0.6, -1.5, -8.0, -0.8, -2.425, -4.4, -0.45],  # C
            [-4.0, -10.6, -4.9, -4.1, -5.9, -4.5, -7.3],  # G
            [-7.8, -1.5, -2.8, -5.0, -4.275, -3.9, -4.65],  # T
            [-3.025, -2.925, -4.1, -4.6, -3.6625, -4.35, -2.975],  # N
            [-5.9, -6.05, -3.85, -4.55, -5.0875, -4.2, -5.975],  # K
            [-0.15, 0.2, -4.35, -4.65, -2.2375, -4.5, 0.025],  # M
        ]
        # GX/CY
        self.delH_mismatch[:, :, 2] = [
            [-2.9, 5.2, -0.6, -8.2, -1.625, -4.4, 1.15],  # A
            [-0.7, 3.6, -9.8, 2.3, -1.15, -3.75, 1.45],  # C
            [0.5, -8.0, -6.0, 3.3, -2.55, -1.35, -3.75],  # G
            [-8.4, 5.2, -4.4, -2.2, -2.45, -3.3, -1.6],  # T
            [-2.875, 1.5, -5.2, -1.2, -1.94375, -3.2, -0.6875],  # N
            [-3.95, -1.4, -5.2, 0.55, -2.5, -2.325, -2.675],  # K
            [-1.8, 4.4, -5.2, -2.95, -1.3875, -4.075, 1.3],  # M
        ]
        # TX/AY
        self.delH_mismatch[:, :, 3] = [
            [4.7, 3.4, 0.7, -7.2, 0.4, -3.25, 4.05],  # A
            [7.6, 6.1, -8.2, 1.2, 1.675, -3.5, 6.85],  # C
            [3.0, -8.5, 1.6, -0.1, -1.0, 0.75, -2.75],  # G
            [-7.6, 1.0, -1.3, 0.2, -1.925, -0.55, -3.3],  # T
            [1.925, 0.5, -1.8, -1.475, -0.2125, -1.625, 1.2125],  # N
            [-2.3, -3.75, 0.15, 0.05, -1.4625, 0.1, -3.025],  # K
            [6.15, 4.75, -3.75, -3.0, 1.0375, -3.375, 5.45],  # M
        ]

        self.delS_mismatch = numpy.zeros((7, 7, 7))
        # AX/TY
        self.delS_mismatch[:, :, 0] = [
            [1.7, 4.6, -2.3, -21.3, -4.325, -11.8, 3.15],  # A
            [14.6, -4.4, -22.4, 0.2, -3.0, -11.1, 5.1],  # C
            [-2.3, -21.0, -9.5, 0.9, -7.975, -4.3, -11.65],  # G
            [-20.4, -6.2, -8.3, -10.8, -11.425, -9.55, -13.3],  # T
            [-1.6, -6.75, -10.625, -7.75, -6.68125, -9.1875, -4.175],  # N
            [-11.35, -13.6, -8.9, -4.95, -9.7, -6.925, -12.475],  # K
            [8.15, 0.1, -12.35, -10.55, -3.6625, -11.45, 4.125],  # M
        ]
        # CX/GY
        self.delS_mismatch[:, :, 1] = [
            [-4.2, 3.7, -2.3, -22.7, -6.375, -12.5, -0.25],  # A
            [-0.6, -7.2, -19.9, -4.5, -8.05, -12.2, -3.9],  # C
            [-13.2, -27.2, -15.3, -11.7, -16.85, -13.5, -20.2],  # G
            [-21.0, -6.1, -8.0, -15.8, -12.725, -11.9, -13.55],  # T
            [-9.75, -9.2, -11.375, -13.675, -11.0, -12.5375, -9.475],  # N
            [-17.1, -16.65, -11.65, -13.75, -14.7875, -12.625, -16.875],  # K
            [-2.4, -1.75, -11.1, -13.6, -7.2125, -12.35, -2.075],  # M
        ]
        # GX/CY
        self.delS_mismatch[:, :, 2] = [
            [-9.8, 14.2, -1.0, -22.2, -4.7, -11.6, 2.2],  # A
            [-3.8, 8.9, -24.4, 5.4, -3.475, -9.5, 2.55],  # C
            [3.2, -19.9, -15.8, 10.4, -5.525, -2.7, -8.35],  # G
            [-22.4, 13.5, -12.3, -8.4, -7.4, -10.35, -4.45],  # T
            [-8.2, 4.175, -13.375, -3.7, -5.275, -8.0375, -2.0125],  # N
            [-9.6, -3.2, -14.05, 1.0, -6.4625, -6.525, -6.4],  # K
            [-6.8, 11.55, -12.7, -8.4, -4.0875, -10.55, 2.375],  # M
        ]
        # TX/AY
        self.delS_mismatch[:, :, 3] = [
            [12.9, 8.0, 0.7, -21.3, 0.075, -10.3, 10.45],  # A
            [20.2, 16.4, -22.2, 0.7, 3.775, -10.75, 18.3],  # C
            [7.4, -22.7, 3.6, -1.7, -3.35, 0.95, -7.65],  # G
            [-21.3, 0.7, -5.3, -1.5, -6.85, -3.4, -10.3],  # T
            [4.8, 0.6, -5.8, -5.95, -1.5875, -5.375, 2.7],  # N
            [-6.95, -11.0, -0.85, -1.6, -5.1, -1.225, -8.975],  # K
            [16.55, 12.2, -10.75, -10.3, 1.925, -10.525, 14.375],  # M
        ]

        self.delG_mismatch = numpy.zeros((7, 7, 7))
        # AX/TY
        self.delG_mismatch[:, :, 0] = [
            [0.61, 0.88, 0.14, -1.00, 0.1575, -0.43, 0.745],  # A
            [0.77, 1.33, -1.44, 0.64, 0.325, -0.4, 1.05],  # C
            [0.02, -1.28, -0.13, 0.71, -0.17, 0.29, -0.63],  # G
            [-0.88, 0.73, 0.07, 0.69, 0.1525, 0.38, -0.075],  # T
            [0.13, 0.415, -0.34, 0.2675, 0.116, -0.04, 0.2725],  # N
            [-0.43, -0.275, 0.03, 0.7, 0.00625, 0.335, -0.3525],  # K
            [0.69, 1.105, -1.44, 0.82, 0.24125, -0.415, 0.8975],  # M
        ]
        # CX/GY
        self.delG_mismatch[:, :, 1] = [
            [0.43, 0.75, 0.03, -1.45, -0.06, -0.71, 0.59],  # A
            [0.79, 0.70, -1.84, 0.62, 0.0675, -0.61, 0.745],  # C
            [0.11, -2.17, -0.11, -0.47, -0.66, -0.29, -1.03],  # G
            [-1.28, 0.40, -0.32, -0.13, -0.3325, -0.225, -0.44],  # T
            [0.0125, -0.08, -0.575, -0.3575, -0.25, -0.425, -0.03375],  # N
            [-0.585, -0.885, -0.215, -0.3, -0.49625, -0.2575, -0.735],  # K
            [0.61, 0.725, -0.905, -0.415, 0.00375, -0.66, 0.6675],  # M
        ]
        # GX/CY
        self.delG_mismatch[:, :, 2] = [
            [0.17, 0.81, -0.25, -1.30, -0.1425, -0.775, 0.49],  # A
            [0.47, 0.79, -2.24, 0.62, -0.09, -0.81, 0.63],  # C
            [-0.52, -1.84, -1.11, 0.08, -0.8475, -0.515, -1.18],  # G
            [-1.44, 0.98, -0.59, 0.45, -0.15, -0.07, -0.23],  # T
            [-0.33, 0.185, -1.0475, -0.0375, -0.31, -0.5575, -0.0725],  # N
            [-0.98, -0.43, -0.85, 0.265, -0.49875, -0.2925, -0.705],  # K
            [0.32, 0.8, -1.245, 0.34, -0.01625, -0.7925, 0.56],  # M
        ]
        # TX/AY
        self.delG_mismatch[:, :, 3] = [
            [0.69, 0.92, 0.42, -0.58, 0.3625, -0.08, 0.805],  # A
            [1.33, 1.05, -1.30, 0.97, 0.5125, -0.165, 1.19],  # C
            [0.74, -1.45, 0.44, 0.43, 0.04, 0.435, -0.355],  # G
            [-1.00, 0.75, 0.34, 0.68, 0.1925, 0.51, -0.125],  # T
            [0.44, 0.3175, -0.025, 0.375, 0.27688, 0.175, 0.37875],  # N
            [-0.13, -0.35, 0.39, 0.555, 0.11625, 0.4725, -0.24],  # K
            [1.01, 0.985, -0.44, 0.195, 0.4375, -0.1225, 0.9975],  # M
        ]

        self.delH_init = 0.2
        self.delS_init = -5.7


Nearest_Neighbor = Nearest_Neighbor_Parameter()


def _convert_sequence(sequence):
    # Easier to keep track of integers in Matlab
    # A,C,G,T,N,K,M --> 1,2,3,4,5,6,7.
    sequence = sequence.upper()
    numerical_sequence = numpy.zeros((1, len(sequence)), dtype=numpy.int16)
    seq2num_dict = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 4, "K": 5, "M": 6}

    for i, seq in enumerate(sequence):
        numerical_sequence[0, i] = seq2num_dict[seq]
    return numerical_sequence


def _ionic_strength_correction(Tm, monovalent_conc, divalent_conc, f_GC, N_BP):
    # From Owczarzy et al., Biochemistry, 2008.
    R = math.sqrt(divalent_conc) / monovalent_conc
    Tm_corrected = Tm
    if R < 0.22:
        # Monovalent dominated
        x = math.log(monovalent_conc)
        Tm_corrected = 1.0 / (
            (1.0 / Tm) + (4.29 * f_GC - 3.95) * 1e-5 * x + 9.40e-6 * math.pow(x, 2)
        )
    else:
        # Divalent dominated
        (a, b, c, d, e, f, g) = (
            3.92e-5,
            -9.11e-6,
            6.26e-5,
            1.42e-5,
            -4.82e-4,
            5.25e-4,
            8.31e-5,
        )

        if R < 6.0:
            # Some competition from monovalent
            y = monovalent_conc
            a *= 0.843 - 0.352 * math.sqrt(y) * math.log(y)
            d *= 1.279 - 4.03e-3 * math.log(y) - 8.03e-3 * math.pow(math.log(y), 2)
            g *= 0.486 - 0.258 * math.log(y) + f * 10 * math.pow(math.log(y), 3)

        x = math.log(divalent_conc)
        Tm_corrected = 1.0 / (
            (1.0 / Tm)
            + a
            + b * x
            + f_GC * (c + d * x)
            + (1.0 / (2 * (N_BP - 1))) * (e + f * x + g * math.pow(x, 2))
        )
    return Tm_corrected


def _precalculate_Tm(
    sequence, DNA_conc=0.2e-6, monovalent_conc=0.1, divalent_conc=0.0015
) -> float:
    # This could be sped up significantly, since many of the sums of
    # delH, delG are shared between calculations.
    numerical_sequence = _convert_sequence(sequence)
    delS_DNA_conc = 1.987 * math.log(DNA_conc / 2)
    delS_init = Nearest_Neighbor.delS_init + delS_DNA_conc
    N_BP = len(sequence)

    delH_matrix = Nearest_Neighbor.delH_init * numpy.ones((N_BP, N_BP))
    delS_matrix = delS_init * numpy.ones((N_BP, N_BP))
    f_GC = numpy.zeros((N_BP, N_BP))
    len_BP = numpy.ones((N_BP, N_BP))

    for i in range(N_BP):
        if numerical_sequence[0, i] in (1, 2):
            f_GC[i, i] = 1

    print("Filling delH, delS matrix ...")
    for i in range(N_BP):
        for j in range(i + 1, N_BP):
            delH_matrix[i, j] = (
                delH_matrix[i, j - 1]
                + Nearest_Neighbor.delH_NN[
                    numerical_sequence[0, j - 1], numerical_sequence[0, j]
                ]
            )
            delS_matrix[i, j] = (
                delS_matrix[i, j - 1]
                + Nearest_Neighbor.delS_NN[
                    numerical_sequence[0, j - 1], numerical_sequence[0, j]
                ]
            )
            len_BP[i, j] = len_BP[i, j - 1] + 1

            f_GC[i, j] = f_GC[i, j - 1]
            if numerical_sequence[0, j] in (1, 2):
                f_GC[i, j] += 1

    print("Terminal penalties ...")
    for i in range(N_BP):
        for j in range(i + 1, N_BP):
            delH_matrix[i, j] += Nearest_Neighbor.delH_AT_closing_penalty[
                numerical_sequence[0, i]
            ]
            delH_matrix[i, j] += Nearest_Neighbor.delH_AT_closing_penalty[
                numerical_sequence[0, j]
            ]

            delS_matrix[i, j] += Nearest_Neighbor.delS_AT_closing_penalty[
                numerical_sequence[0, i]
            ]
            delS_matrix[i, j] += Nearest_Neighbor.delS_AT_closing_penalty[
                numerical_sequence[0, j]
            ]

    Tm = 1000 * numpy.divide(delH_matrix, delS_matrix)
    f_GC = numpy.divide(f_GC, len_BP)

    print("Ionic strength corrections ...")
    for i in range(N_BP):
        for j in range(i, N_BP):
            Tm[i, j] = _ionic_strength_correction(
                Tm[i, j], monovalent_conc, divalent_conc, f_GC[i, j], len_BP[i, j]
            )

    return Tm - 273.15


def calc_Tm(sequence, DNA_conc=1e-5, monovalent_conc=1.0, divalent_conc=0.0):
    """Calculate melting temperature for a given sequence

    Args:
        sequence: ``str``: Annealing DNA sequence section.
        DNA_conc: ``float``: `(Optional)` Concentration of DNA.
        monovalent_conc: ``float``: `(Optional)` Monovalent cation concentration.
        divalent_conc: ``float``: `(Optional)` Divalent cation concentration.

    Returns:
        ``float``: Melting temperature in Celcius
    """

    numerical_sequence = _convert_sequence(sequence)
    delS_DNA_conc = 1.987 * math.log(DNA_conc / 2)
    delS_sum = Nearest_Neighbor.delS_init + delS_DNA_conc
    delH_sum = Nearest_Neighbor.delH_init
    N_BP = len(sequence)

    for i in range(N_BP - 1):
        delH_sum += Nearest_Neighbor.delH_NN[
            numerical_sequence[0, i], numerical_sequence[0, i + 1]
        ]
        delS_sum += Nearest_Neighbor.delS_NN[
            numerical_sequence[0, i], numerical_sequence[0, i + 1]
        ]

    delH_sum += Nearest_Neighbor.delH_AT_closing_penalty[numerical_sequence[0, 0]]
    delH_sum += Nearest_Neighbor.delH_AT_closing_penalty[numerical_sequence[0, -1]]
    delS_sum += Nearest_Neighbor.delS_AT_closing_penalty[numerical_sequence[0, 0]]
    delS_sum += Nearest_Neighbor.delS_AT_closing_penalty[numerical_sequence[0, -1]]

    Tm = 1000 * delH_sum / delS_sum
    f_GC = (
        numpy.sum(numerical_sequence == 1) + numpy.sum(numerical_sequence == 2)
    ) / float(N_BP)
    Tm = _ionic_strength_correction(Tm, monovalent_conc, divalent_conc, f_GC, N_BP)
    return Tm - 273.15
