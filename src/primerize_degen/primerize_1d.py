import argparse
import math
import numpy as np
import time
import traceback
from typing import List, Dict, Optional, Union, Tuple

from . import misprime
from . import thermo
from . import util
from . import util_class
from .wrapper import Design_Single


try:
    from numba import jit
except ImportError:

    def jit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]

        def decorator(func):
            return func

        return decorator


class Primerize_1D(thermo.Singleton):
    """Construct a worker for 1D Primer Design (Simple Assembly).

    Args:
        MIN_TM: ``float``: `(Optional)` Minimum annealing temperature for overlapping regions. Unit in Celsius. Positive number only.
        NUM_PRIMERS: ``int``: `(Optional)` Exact limit of number of primers in design. Non-negative even number only. `0` represents "No limit".
        MIN_LENGTH: ``int``: `(Optional)` Minimum length allowed for each primer. Positive number only.
        MAX_LENGTH: ``int``: `(Optional)` Maximum length allowed for each primer. Positive number only.
        COL_SIZE: ``int``: `(Optional)` Column width for assembly output. Positive number only.
        WARN_CUTOFF: ``int``: `(Optional)` Threshold of pairing region length for misprime warning. Positive number only.
        prefix: ``str``: `(Optional)` Construct prefix/name.

    Returns:
        ``primerize.Primerize_1D``

    Note:
        This ``class`` follows the singleton pattern so that only one instance is created. An instance is already initialized as ``primerize.Primerize_1D``.
    """

    def __init__(
        self,
        MIN_TM: float = 60.0,
        NUM_PRIMERS: int = 0,
        MIN_LENGTH: int = 15,
        MAX_LENGTH: int = 60,
        COL_SIZE: int = 142,
        WARN_CUTOFF: int = 3,
        prefix: str = "primer",
    ):
        self.prefix: str = prefix
        self.MIN_TM: float = max(MIN_TM, 0)
        self.NUM_PRIMERS: int = max(NUM_PRIMERS, 0)
        self.MIN_LENGTH: int = max(MIN_LENGTH, 0)
        self.MAX_LENGTH: int = max(MAX_LENGTH, 0)
        self.COL_SIZE: int = max(COL_SIZE, 0)
        self.WARN_CUTOFF: int = max(WARN_CUTOFF, 0)

    def __repr__(self):
        """Representation of the ``Primerize_1D`` class."""

        return repr(self.__dict__)

    def __str__(self):
        return repr(self.__dict__)

    def get(self, key):
        """Get current worker parameters.

        Args:
            key: ``str``: Keyword of parameter. Valid keywords are ``'MIN_TM'``, ``'NUM_PRIMERS'``, ``'MIN_LENGTH'``, ``'MAX_LENGTH'``, ``'prefix'``; case insensitive.

        Returns:
            value of specified **key**.

        Raises:
            AttributeError: For illegal **key**.
        """

        key = key.upper()
        if hasattr(self, key):
            return getattr(self, key)
        elif key == "PREFIX":
            return self.prefix
        else:
            raise AttributeError(
                "\033[41mERROR\033[0m: Unrecognized key \033[92m%s\033[0m for \033[94m%s.get()\033[0m.\n"
                % (key, self.__class__)
            )

    def set(self, key, value):
        """Set current worker parameters.

        Args:
            key: ``str``: Keyword of parameter. Valid keywords are the same as ``get()``.
            value: ``(auto)``: New value for specified keyword. Type of value must match **key**.

        Raises:
            AttributeError: For illegal **key**.
            ValueError: For illegal **value**.
        """

        key = key.upper()
        if hasattr(self, key) or key == "PREFIX":
            if key == "PREFIX":
                self.prefix = str(value)
            elif isinstance(value, (int, float)) and value > 0:
                if key == "MIN_TM":
                    self.MIN_TM = float(value)
                else:
                    setattr(self, key, int(value))
                    if self.MIN_LENGTH > self.MAX_LENGTH:
                        print(
                            "\033[93mWARNING\033[0m: \033[92mMIN_LENGTH\033[0m is greater than \033[92mMAX_LENGTH\033[0m."
                        )
                    elif self.NUM_PRIMERS % 2:
                        print(
                            "\033[93mWARNING\033[0m: \033[92mNUM_PRIMERS\033[0m should be even number only."
                        )
            else:
                raise ValueError(
                    "\033[41mERROR\033[0m: Illegal value \033[95m%s\033[0m for key \033[92m%s\033[0m for \033[94m%s.set()\033[0m.\n"
                    % (value, key, self.__class__)
                )
        else:
            raise AttributeError(
                "\033[41mERROR\033[0m: Unrecognized key \033[92m%s\033[0m for \033[94m%s.set()\033[0m.\n"
                % (key, self.__class__)
            )

    def reset(self):
        """Reset current worker parameters to default."""

        self.prefix = "primer"
        self.MIN_TM = 60.0
        self.NUM_PRIMERS = 0
        self.MIN_LENGTH = 15
        self.MAX_LENGTH = 60
        self.COL_SIZE = 142
        self.WARN_CUTOFF = 3

    def design(
        self,
        sequence,
        MIN_TM=None,
        NUM_PRIMERS=None,
        MIN_LENGTH=None,
        MAX_LENGTH=None,
        prefix=None,
    ) -> Design_Single:
        """Run design code to get a PCR Assembly solution for input sequence under specified conditions. Current worker parameters are used for nonspecified optional arguments.

        Args:
            sequence: ``str``: Sequence for assembly design. Valid RNA/DNA sequence only, case insensitive.
            MIN_TM: ``float``: `(Optional)` Minimum annealing temperature for overlapping regions.
            NUM_PRIMERS: ``int``: `(Optional)` Exact limit of number of primers in design.
            MIN_LENGTH: ``int``: `(Optional)` Minimum length allowed for each primer.
            MAX_LENGTH: ``int``: `(Optional)` Maximum length allowed for each primer.
            prefix: ``str``: `(Optional)` Construct prefix/name.

        Returns:
            ``primerize.Design_Single``
        """

        MIN_TM: float = self.MIN_TM if MIN_TM is None else MIN_TM
        NUM_PRIMERS: int = self.NUM_PRIMERS if NUM_PRIMERS is None else NUM_PRIMERS
        MIN_LENGTH: int = self.MIN_LENGTH if MIN_LENGTH is None else MIN_LENGTH
        MAX_LENGTH: int = self.MAX_LENGTH if MAX_LENGTH is None else MAX_LENGTH
        prefix: str = self.prefix if prefix is None else prefix

        name: str = prefix
        sequence: str = util.RNA2DNA(sequence)
        numerical_sequence = thermo._convert_sequence(sequence)
        N_BP: int = len(sequence)
        params: Dict[str, Optional[Union[int, float, str]]] = {
            "MIN_TM": MIN_TM,
            "NUM_PRIMERS": NUM_PRIMERS,
            "MIN_LENGTH": MIN_LENGTH,
            "MAX_LENGTH": MAX_LENGTH,
            "N_BP": N_BP,
            "COL_SIZE": self.COL_SIZE,
            "WARN_CUTOFF": self.WARN_CUTOFF,
        }

        is_success: bool = True
        primers: List[str] = list()
        warnings: List[str] = list()
        assembly = {}
        misprime_score = ["", ""]

        try:
            print("Precalculataing Tm matrix ...")
            Tm_precalculated: float = thermo._precalculate_Tm(sequence)
            print("Precalculataing misprime score ...")
            (
                num_match_forward,
                num_match_reverse,
                best_match_forward,
                best_match_reverse,
                misprime_score_forward,
                misprime_score_reverse,
            ) = misprime._check_misprime(sequence)

            print("Doing dynamics programming calculation ...")
            (
                scores_start,
                scores_stop,
                scores_final,
                choice_start_p,
                choice_start_q,
                choice_stop_i,
                choice_stop_j,
                MAX_SCORE,
                N_primers,
            ) = _dynamic_programming(
                NUM_PRIMERS,
                MIN_LENGTH,
                MAX_LENGTH,
                MIN_TM,
                N_BP,
                numerical_sequence,
                misprime_score_forward,
                misprime_score_reverse,
                Tm_precalculated,
            )
            print("Doing backtracking ...")
            (is_success, primers, primer_set, warnings) = _back_tracking(
                N_BP,
                sequence,
                scores_final,
                choice_start_p,
                choice_start_q,
                choice_stop_i,
                choice_stop_j,
                N_primers,
                MAX_SCORE,
                num_match_forward,
                num_match_reverse,
                best_match_forward,
                best_match_reverse,
                self.WARN_CUTOFF,
            )

            if is_success:
                allow_forward_line = list(" " * N_BP)
                allow_reverse_line = list(" " * N_BP)
                for i in range(N_BP):
                    allow_forward_line[i] = str(
                        int(min(num_match_forward[0, i] + 1, 9))
                    )
                    allow_reverse_line[i] = str(
                        int(min(num_match_reverse[0, i] + 1, 9))
                    )

                misprime_score = [
                    "".join(allow_forward_line).strip(),
                    "".join(allow_reverse_line).strip(),
                ]
                assembly = util_class.Assembly(sequence, primers, name, self.COL_SIZE)
                print("\033[92mSUCCESS\033[0m: Primerize 1D design() finished.\n")
            else:
                print(
                    "\033[41mFAIL\033[0m: \033[41mNO Solution\033[0m found under given contraints.\n"
                )
        except:
            is_success = False
            print(traceback.format_exc())
            print("\033[41mERROR\033[0m: Primerize 1D design() encountered error.\n")
            data = {"misprime_score": [], "assembly": [], "warnings": []}
            return Design_Single(
                {
                    "sequence": sequence,
                    "name": name,
                    "is_success": is_success,
                    "primer_set": [],
                    "params": params,
                    "data": data,
                }
            )

        data = {
            "misprime_score": misprime_score,
            "assembly": assembly,
            "warnings": warnings,
        }
        return Design_Single(
            {
                "sequence": sequence,
                "name": name,
                "is_success": is_success,
                "primer_set": primer_set,
                "params": params,
                "data": data,
            }
        )


@jit(nopython=True, nogil=True, cache=False)
def _check_overlap_region(sequence: np.ndarray, start: int, end: int) -> bool:
    """Helper function to check if a region contains degenerate nucleotides"""
    for i in range(start, end):
        if sequence[0, i] >= 4:  # N=4, K=5, M=6
            return False
    return True


@jit(nopython=True, nogil=True, cache=False)
def _dynamic_programming(
    NUM_PRIMERS: int,
    MIN_LENGTH: int,
    MAX_LENGTH: int,
    MIN_TM: float,
    N_BP: int,
    numerical_sequence: np.ndarray,
    misprime_score_forward: np.ndarray,
    misprime_score_reverse: np.ndarray,
    Tm_precalculated: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
]:
    # could be zero, meaning user does not know.
    num_primer_sets: int = int(NUM_PRIMERS / 2)
    num_primer_sets_max: int = int(math.ceil(N_BP / float(MIN_LENGTH)))

    misprime_score_weight: float = 10.0
    MAX_SCORE: int = N_BP * 2 + 1
    MAX_SCORE += (
        misprime_score_weight
        * max(np.amax(misprime_score_forward), np.amax(misprime_score_reverse))
        * 2
        * num_primer_sets_max
    )

    scores_start: np.ndarray = MAX_SCORE * np.ones((N_BP, N_BP, num_primer_sets_max))
    scores_stop: np.ndarray = MAX_SCORE * np.ones((N_BP, N_BP, num_primer_sets_max))
    scores_final: np.ndarray = MAX_SCORE * np.ones((N_BP, N_BP, num_primer_sets_max))

    # used for backtracking:
    choice_start_p: np.ndarray = np.zeros(
        (N_BP, N_BP, num_primer_sets_max), dtype=np.int16
    )
    choice_start_q: np.ndarray = np.zeros(
        (N_BP, N_BP, num_primer_sets_max), dtype=np.int16
    )
    choice_stop_i: np.ndarray = np.zeros(
        (N_BP, N_BP, num_primer_sets_max), dtype=np.int16
    )
    choice_stop_j: np.ndarray = np.zeros(
        (N_BP, N_BP, num_primer_sets_max), dtype=np.int16
    )

    # basic setup -- first primer
    for p in range(MIN_LENGTH, MAX_LENGTH + 1):
        q_min: int = max(1, p - MAX_LENGTH + 1)
        q_max: int = p

        for q in range(q_min, q_max + 1):
            if (
                _check_overlap_region(numerical_sequence, q - 1, p)
                and Tm_precalculated[q - 1, p - 1] > MIN_TM
            ):
                scores_stop[p - 1, q - 1, 0] = (q - 1) + 2 * (p - q + 1)
                scores_stop[p - 1, q - 1, 0] += misprime_score_weight * (
                    misprime_score_forward[0, p - 1] + misprime_score_reverse[0, q - 1]
                )

    best_min_score: int = MAX_SCORE
    n: int = 1
    while n <= num_primer_sets_max:
        # final scoring
        for p in range(1, N_BP + 1):
            q_min: int = max(1, p - MAX_LENGTH + 1)
            q_max: int = p

            for q in range(q_min, q_max + 1):
                if scores_stop[p - 1, q - 1, n - 1] < MAX_SCORE:
                    i: int = N_BP + 1
                    j: int = N_BP
                    last_primer_length: int = j - q + 1
                    if (
                        last_primer_length <= MAX_LENGTH
                        and last_primer_length >= MIN_LENGTH
                    ):
                        scores_final[p - 1, q - 1, n - 1] = scores_stop[
                            p - 1, q - 1, n - 1
                        ] + (i - p - 1)
                        scores_final[p - 1, q - 1, n - 1] += misprime_score_weight * (
                            misprime_score_forward[0, p - 1]
                            + misprime_score_reverse[0, q - 1]
                        )

        min_score: np.ndarray = np.amin(scores_final[:, :, n - 1])
        if min_score < best_min_score or n == 1:
            best_min_score = min_score
            best_n = n

        if n >= num_primer_sets_max:
            break
        if num_primer_sets > 0 and n == num_primer_sets:
            break

        # considering another primer set
        n += 1

        # Check overlaps for internal primers
        for p in range(1, N_BP + 1):
            q_min: int = max(1, p - MAX_LENGTH + 1)
            q_max: int = p

            for q in range(q_min, q_max + 1):
                if scores_stop[p - 1, q - 1, n - 2] < MAX_SCORE:
                    min_j: int = max(p + 1, q + MIN_LENGTH - 1)
                    max_j: int = min(N_BP, q + MAX_LENGTH - 1)

                    for j in range(min_j, max_j + 1):
                        min_i: int = max(p + 1, j - MAX_LENGTH + 1)
                        max_i: int = j

                        for i in range(min_i, max_i + 1):
                            if (
                                _check_overlap_region(numerical_sequence, i - 1, j)
                                and Tm_precalculated[i - 1, j - 1] > MIN_TM
                            ):
                                potential_score = (
                                    scores_stop[p - 1, q - 1, n - 2]
                                    + (i - p - 1)
                                    + 2 * (j - i + 1)
                                )
                                if potential_score < scores_start[i - 1, j - 1, n - 2]:
                                    scores_start[i - 1, j - 1, n - 2] = potential_score
                                    choice_start_p[i - 1, j - 1, n - 2] = p - 1
                                    choice_start_q[i - 1, j - 1, n - 2] = q - 1

        # Check final set of overlaps
        for j in range(1, N_BP + 1):
            min_i: int = max(1, j - MAX_LENGTH + 1)
            max_i: int = j

            for i in range(min_i, max_i + 1):
                if scores_start[i - 1, j - 1, n - 2] < MAX_SCORE:
                    min_p = max(j + 1, i + MIN_LENGTH - 1)
                    max_p = min(N_BP, i + MAX_LENGTH - 1)

                    for p in range(min_p, max_p + 1):
                        min_q = max(j + 1, p - MAX_LENGTH + 1)
                        max_q = p

                        for q in range(min_q, max_q + 1):
                            if (
                                _check_overlap_region(numerical_sequence, q - 1, p)
                                and Tm_precalculated[q - 1, p - 1] > MIN_TM
                            ):
                                potential_score = (
                                    scores_start[i - 1, j - 1, n - 2]
                                    + (q - j - 1)
                                    + 2 * (p - q + 1)
                                )
                                potential_score += misprime_score_weight * (
                                    misprime_score_forward[0, p - 1]
                                    + misprime_score_reverse[0, q - 1]
                                )
                                if potential_score < scores_stop[p - 1, q - 1, n - 1]:
                                    scores_stop[p - 1, q - 1, n - 1] = potential_score
                                    choice_stop_i[p - 1, q - 1, n - 1] = i - 1
                                    choice_stop_j[p - 1, q - 1, n - 1] = j - 1

    if num_primer_sets > 0:
        N_primers = num_primer_sets
    else:
        N_primers = best_n
    return (
        scores_start,
        scores_stop,
        scores_final,
        choice_start_p,
        choice_start_q,
        choice_stop_i,
        choice_stop_j,
        MAX_SCORE,
        N_primers,
    )


def _back_tracking(
    N_BP,
    sequence,
    scores_final,
    choice_start_p,
    choice_start_q,
    choice_stop_i,
    choice_stop_j,
    N_primers,
    MAX_SCORE,
    num_match_forward,
    num_match_reverse,
    best_match_forward,
    best_match_reverse,
    WARN_CUTOFF,
):
    y: np.array = np.amin(scores_final[:, :, N_primers - 1], axis=0)
    idx = np.argmin(scores_final[:, :, N_primers - 1], axis=0)
    min_scroe = np.amin(y)
    q = np.argmin(y)
    p = idx[q]

    is_success = True
    primer_set = []
    misprime_warn = []
    primers = np.zeros((3, 2 * N_primers))
    if min_scroe == MAX_SCORE:
        is_success = False
    else:
        primers[:, 2 * N_primers - 1] = [q, N_BP - 1, -1]
        for m in range(N_primers - 1, 0, -1):
            i = choice_stop_i[p, q, m]
            j = choice_stop_j[p, q, m]
            primers[:, 2 * m] = [i, p, 1]
            p = choice_start_p[i, j, m - 1]
            q = choice_start_q[i, j, m - 1]
            primers[:, 2 * m - 1] = [q, j, -1]
        primers[:, 0] = [0, p, 1]
        primers = primers.astype(int)

        for i in range(2 * N_primers):
            primer_seq = sequence[primers[0, i] : primers[1, i] + 1]
            if primers[2, i] == -1:
                primer_set.append(util.reverse_complement(primer_seq))

                # mispriming "report"
                end_pos = primers[0, i]
                if num_match_reverse[0, end_pos] >= WARN_CUTOFF:
                    problem_primer = _find_primers_affected(
                        primers, best_match_reverse[0, end_pos]
                    )
                    misprime_warn.append(
                        (
                            i + 1,
                            int(num_match_reverse[0, end_pos] + 1),
                            int(best_match_reverse[0, end_pos] + 1),
                            problem_primer,
                        )
                    )
            else:
                primer_set.append(str(primer_seq))

                # mispriming "report"
                end_pos = primers[1, i]
                if num_match_forward[0, end_pos] >= WARN_CUTOFF:
                    problem_primer = _find_primers_affected(
                        primers, best_match_forward[0, end_pos]
                    )
                    misprime_warn.append(
                        (
                            i + 1,
                            int(num_match_forward[0, end_pos] + 1),
                            int(best_match_forward[0, end_pos] + 1),
                            problem_primer,
                        )
                    )

    return (is_success, primers, primer_set, misprime_warn)


def _find_primers_affected(primers, pos):
    primer_list = []
    for i in range(primers.shape[1]):
        if pos >= primers[0, i] and pos <= primers[1, i]:
            primer_list.append(i + 1)
    return primer_list


def main():
    parser = argparse.ArgumentParser(
        description="\033[92mPrimerize 1D PCR Assembly Design\033[0m",
        epilog="\033[94mby Siqi Tian, 2016\033[0m",
        add_help=False,
    )
    parser.add_argument("sequence", type=str, help="DNA Template Sequence")
    parser.add_argument(
        "-p",
        metavar="prefix",
        type=str,
        help="Display Name of Construct",
        dest="prefix",
        default="primer",
    )
    group1 = parser.add_argument_group("advanced options")
    group1.add_argument(
        "-t",
        metavar="MIN_TM",
        type=float,
        help="Minimum Annealing Temperature",
        dest="MIN_TM",
        default=60.0,
    )
    group1.add_argument(
        "-n",
        metavar="NUM_PRIMERS",
        type=int,
        help="Number of Primers",
        dest="NUM_PRIMERS",
        default=0,
    )
    group1.add_argument(
        "-l",
        metavar="MIN_LENGTH",
        type=int,
        help="Minimum Length of each Primer",
        dest="MIN_LENGTH",
        default=15,
    )
    group1.add_argument(
        "-u",
        metavar="MAX_LENGTH",
        type=int,
        help="Maximum Length of each Primer",
        dest="MAX_LENGTH",
        default=60,
    )
    group2 = parser.add_argument_group("commandline options")
    group2.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        dest="is_quiet",
        help="Suppress Results Printing to stdout",
    )
    group2.add_argument(
        "-f",
        "--file",
        action="store_true",
        dest="is_file",
        help="Write Results to Text File",
    )
    group2.add_argument("-h", "--help", action="help", help="Show this Help Message")
    args = parser.parse_args()

    t0 = time.time()
    prm = Primerize_1D()
    res = prm.design(
        args.sequence,
        args.MIN_TM,
        args.NUM_PRIMERS,
        args.MIN_LENGTH,
        args.MAX_LENGTH,
        args.prefix,
    )
    if res.is_success:
        if not args.is_quiet:
            print(res)
        if args.is_file:
            res.save()
    print("Time elapsed: %.1f s." % (time.time() - t0))


if __name__ == "__main__":
    main()
