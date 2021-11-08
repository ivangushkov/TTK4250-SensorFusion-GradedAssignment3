import pickle
from numpy.core.numeric import isscalar
import pytest
from copy import deepcopy
import sys
from pathlib import Path
import numpy as np
import os
from dataclasses import is_dataclass, astuple
from collections.abc import Iterable

assignment_name = "slam"

this_file = Path(__file__)
tests_folder = this_file.parent
test_data_file = tests_folder.joinpath("test_data.pickle")
project_folder = tests_folder.parent
code_folder = project_folder.joinpath(assignment_name)

sys.path.insert(0, str(code_folder))

import solution  # nopep8
import EKFSLAM  # nopep8


@pytest.fixture
def test_data():
    with open(test_data_file, "rb") as file:
        test_data = pickle.load(file)
    return test_data


def compare(a, b):
    if isinstance(b, np.ndarray) or np.isscalar(b):
        if isinstance(b, np.ndarray) and a.shape != b.shape:
            return False
        return np.allclose(a, b, atol=1e-6)

    elif is_dataclass(b):
        if type(a).__name__ != type(b).__name__:
            return False
        a_tup, b_tup = astuple(a), astuple(b)
        return all([compare(i, j) for i, j in zip(a_tup, b_tup)])

    elif isinstance(b, Iterable):
        return all([compare(i, j) for i, j in zip(a, b)])

    else:
        return a == b


class Test_EKFSLAM_f:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["EKFSLAM.EKFSLAM.f"]:
            params = tuple(finput.values())

            self_1, x_1, u_1 = deepcopy(params)

            self_2, x_2, u_2 = deepcopy(params)

            xpred_1 = EKFSLAM.EKFSLAM.f(self_1, x_1, u_1)

            xpred_2 = solution.EKFSLAM.EKFSLAM.f(self_2, x_2, u_2)
            
            assert compare(xpred_1, xpred_2)
            
            assert compare(self_1, self_2)
            assert compare(x_1, x_2)
            assert compare(u_1, u_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["EKFSLAM.EKFSLAM.f"][:1]:
            params = finput

            solution.used["EKFSLAM.EKFSLAM.f"] = False

            EKFSLAM.EKFSLAM.f(**params)

            assert not solution.used["EKFSLAM.EKFSLAM.f"], "The function uses the solution"


class Test_EKFSLAM_Fx:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["EKFSLAM.EKFSLAM.Fx"]:
            params = tuple(finput.values())

            self_1, x_1, u_1 = deepcopy(params)

            self_2, x_2, u_2 = deepcopy(params)

            Fx_1 = EKFSLAM.EKFSLAM.Fx(self_1, x_1, u_1)

            Fx_2 = solution.EKFSLAM.EKFSLAM.Fx(self_2, x_2, u_2)
            
            assert compare(Fx_1, Fx_2)
            
            assert compare(self_1, self_2)
            assert compare(x_1, x_2)
            assert compare(u_1, u_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["EKFSLAM.EKFSLAM.Fx"][:1]:
            params = finput

            solution.used["EKFSLAM.EKFSLAM.Fx"] = False

            EKFSLAM.EKFSLAM.Fx(**params)

            assert not solution.used["EKFSLAM.EKFSLAM.Fx"], "The function uses the solution"


class Test_EKFSLAM_Fu:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["EKFSLAM.EKFSLAM.Fu"]:
            params = tuple(finput.values())

            self_1, x_1, u_1 = deepcopy(params)

            self_2, x_2, u_2 = deepcopy(params)

            Fu_1 = EKFSLAM.EKFSLAM.Fu(self_1, x_1, u_1)

            Fu_2 = solution.EKFSLAM.EKFSLAM.Fu(self_2, x_2, u_2)
            
            assert compare(Fu_1, Fu_2)
            
            assert compare(self_1, self_2)
            assert compare(x_1, x_2)
            assert compare(u_1, u_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["EKFSLAM.EKFSLAM.Fu"][:1]:
            params = finput

            solution.used["EKFSLAM.EKFSLAM.Fu"] = False

            EKFSLAM.EKFSLAM.Fu(**params)

            assert not solution.used["EKFSLAM.EKFSLAM.Fu"], "The function uses the solution"


class Test_EKFSLAM_predict:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["EKFSLAM.EKFSLAM.predict"]:
            params = tuple(finput.values())

            self_1, eta_1, P_1, z_odo_1 = deepcopy(params)

            self_2, eta_2, P_2, z_odo_2 = deepcopy(params)

            etapred_1, P_1 = EKFSLAM.EKFSLAM.predict(self_1, eta_1, P_1, z_odo_1)

            etapred_2, P_2 = solution.EKFSLAM.EKFSLAM.predict(self_2, eta_2, P_2, z_odo_2)
            
            assert compare(etapred_1, etapred_2)
            assert compare(P_1, P_2)
            
            assert compare(self_1, self_2)
            assert compare(eta_1, eta_2)
            assert compare(P_1, P_2)
            assert compare(z_odo_1, z_odo_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["EKFSLAM.EKFSLAM.predict"][:1]:
            params = finput

            solution.used["EKFSLAM.EKFSLAM.predict"] = False

            EKFSLAM.EKFSLAM.predict(**params)

            assert not solution.used["EKFSLAM.EKFSLAM.predict"], "The function uses the solution"


class Test_EKFSLAM_h:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["EKFSLAM.EKFSLAM.h"]:
            params = tuple(finput.values())

            self_1, eta_1 = deepcopy(params)

            self_2, eta_2 = deepcopy(params)

            zpred_1 = EKFSLAM.EKFSLAM.h(self_1, eta_1)

            zpred_2 = solution.EKFSLAM.EKFSLAM.h(self_2, eta_2)
            
            assert compare(zpred_1, zpred_2)
            
            assert compare(self_1, self_2)
            assert compare(eta_1, eta_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["EKFSLAM.EKFSLAM.h"][:1]:
            params = finput

            solution.used["EKFSLAM.EKFSLAM.h"] = False

            EKFSLAM.EKFSLAM.h(**params)

            assert not solution.used["EKFSLAM.EKFSLAM.h"], "The function uses the solution"


class Test_EKFSLAM_h_jac:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["EKFSLAM.EKFSLAM.h_jac"]:
            params = tuple(finput.values())

            self_1, eta_1 = deepcopy(params)

            self_2, eta_2 = deepcopy(params)

            H_1 = EKFSLAM.EKFSLAM.h_jac(self_1, eta_1)

            H_2 = solution.EKFSLAM.EKFSLAM.h_jac(self_2, eta_2)
            
            assert compare(H_1, H_2)
            
            assert compare(self_1, self_2)
            assert compare(eta_1, eta_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["EKFSLAM.EKFSLAM.h_jac"][:1]:
            params = finput

            solution.used["EKFSLAM.EKFSLAM.h_jac"] = False

            EKFSLAM.EKFSLAM.h_jac(**params)

            assert not solution.used["EKFSLAM.EKFSLAM.h_jac"], "The function uses the solution"


class Test_EKFSLAM_add_landmarks:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["EKFSLAM.EKFSLAM.add_landmarks"]:
            params = tuple(finput.values())

            self_1, eta_1, P_1, z_1 = deepcopy(params)

            self_2, eta_2, P_2, z_2 = deepcopy(params)

            etaadded_1, Padded_1 = EKFSLAM.EKFSLAM.add_landmarks(self_1, eta_1, P_1, z_1)

            etaadded_2, Padded_2 = solution.EKFSLAM.EKFSLAM.add_landmarks(self_2, eta_2, P_2, z_2)
            
            assert compare(etaadded_1, etaadded_2)
            assert compare(Padded_1, Padded_2)
            
            assert compare(self_1, self_2)
            assert compare(eta_1, eta_2)
            assert compare(P_1, P_2)
            assert compare(z_1, z_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["EKFSLAM.EKFSLAM.add_landmarks"][:1]:
            params = finput

            solution.used["EKFSLAM.EKFSLAM.add_landmarks"] = False

            EKFSLAM.EKFSLAM.add_landmarks(**params)

            assert not solution.used["EKFSLAM.EKFSLAM.add_landmarks"], "The function uses the solution"


class Test_EKFSLAM_update:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["EKFSLAM.EKFSLAM.update"]:
            params = tuple(finput.values())

            self_1, eta_1, P_1, z_1 = deepcopy(params)

            self_2, eta_2, P_2, z_2 = deepcopy(params)

            etaupd_1, Pupd_1, NIS_1, a_1 = EKFSLAM.EKFSLAM.update(self_1, eta_1, P_1, z_1)

            etaupd_2, Pupd_2, NIS_2, a_2 = solution.EKFSLAM.EKFSLAM.update(self_2, eta_2, P_2, z_2)
            
            assert compare(etaupd_1, etaupd_2)
            assert compare(Pupd_1, Pupd_2)
            assert compare(NIS_1, NIS_2)
            assert compare(a_1, a_2)
            
            assert compare(self_1, self_2)
            assert compare(eta_1, eta_2)
            assert compare(P_1, P_2)
            assert compare(z_1, z_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["EKFSLAM.EKFSLAM.update"][:1]:
            params = finput

            solution.used["EKFSLAM.EKFSLAM.update"] = False

            EKFSLAM.EKFSLAM.update(**params)

            assert not solution.used["EKFSLAM.EKFSLAM.update"], "The function uses the solution"


if __name__ == "__main__":
    os.environ["_PYTEST_RAISE"] = "1"
    pytest.main()
