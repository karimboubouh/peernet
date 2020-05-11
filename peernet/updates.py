import traceback
import numpy as np
from .constants import CONFIDENCE_MEASURE
from .helpers import log


def mp_update(node, parameter="W"):
    """Calculate model propagation update."""
    try:
        theta = node.model.weights
        theta_sol = node.solitary_model.weights
        sigma = np.zeros(theta.shape)

        for name, model in node.models.copy().items():
            theta_k = model.weights
            sigma += (node.W[name] / node.D) * theta_k

        log("info", f"{node.pname}: Model to be updated (Parameter: {parameter}) {np.sum(theta)}")

        alpha_bar = (1 - node.alpha)
        update = (node.alpha + alpha_bar * node.c) ** -1 * (node.alpha * sigma + alpha_bar * node.c * theta_sol)

        log("info", f"{node.pname}: Model updated (Parameter: {parameter}) {np.sum(update)}")

        if update is None:
            print(f"{node.pname} : update={theta_sol}")
            return
        else:
            assert (theta.shape == update.shape)
            node.model.weights = update
            # print(".", end="")
    except AttributeError as e:
        traceback.print_exc()
        return
    except TypeError as e:
        return


def cmp_update(node, parameter="W"):
    """Calculate controlled model propagation update."""
    try:
        theta = node.model.weights
        theta_sol = node.solitary_model.weights
        sigma = np.zeros(theta.shape)
        if node.use_cf:
            if CONFIDENCE_MEASURE == 'max':
                c = 1 / np.max(list(node.cf.values()))
            else:
                c = 1 / np.mean(list(node.cf.values()))
        else:
            c = 1

        for name, model in node.models.copy().items():
            if name != "w7":
                theta_k = model.weights
                sigma += (node.W[name] / node.D) * theta_k

        log("info", f"{node.pname}: Model to be updated (Parameter: {parameter}) {np.sum(theta)}")

        alpha_bar = (1 - node.alpha)
        update = (node.alpha + alpha_bar * c) ** -1 * (node.alpha * sigma + alpha_bar * c * theta_sol)

        log("info", f"{node.pname}: Model updated (Parameter: {parameter}) {np.sum(update)}")

        if update is None:
            print(f"{node.pname} : update={theta_sol}")
            return
        else:
            assert (theta.shape == update.shape)
            node.model.weights = update
            # print(".", end="")
    except AttributeError as e:
        traceback.print_exc()
        return
    except TypeError as e:
        traceback.print_exc()
        return


def mp_update2(node, parameter="W"):
    """Calculate model propagation update."""
    try:
        theta = getattr(node.model, parameter, None)
        theta_sol = getattr(node.solitary_model, parameter, None)
        if theta is None:
            print(f"{node.pname} : theta={theta}")
        if theta_sol is None:
            print(f"{node.pname} : theta_sol={theta_sol}")
            return
        sigma = np.zeros((theta.shape[0], 1))
        for name, model in node.models.copy().items():
            theta_k = getattr(model, parameter)
            sigma += (node.W[name] / node.D) * theta_k

        log("info", f"{node.pname}: Model to be updated (Parameter: {parameter}) {sum(theta)}")

        alpha_bar = (1 - node.alpha)
        update = (node.alpha + alpha_bar * node.c) ** -1 * (node.alpha * sigma + alpha_bar * node.c * theta_sol)

        log("info", f"{node.pname}: Model updated (Parameter: {parameter}) {sum(update)}")

        if update is None:
            print(f"{node.pname} : update={theta_sol}")
            return
        else:
            setattr(node.model, parameter, update)
            assert (theta.shape == update.shape)
            # if node.stop_condition % 10 == 0 or node.stop_condition < 5:
            #     print(f'({node.name}-{node.stop_condition})', end='')
            # print("*", end=" ")

    except AttributeError as e:
        traceback.print_exc()
        return
    except TypeError as e:
        return


def cl_update_primal(node, neighbor, parameter="W"):
    """Calculate primal variables (minimize arg min L(models, Z, A))."""
    if node.check_exchange():
        neighbor_name = neighbor['name']
        node.Theta[node.name] = node.model.collaborative_fit(node, neighbor=False)
        node.Theta[neighbor_name] = node.model.collaborative_fit(node, neighbor=neighbor_name)
        # Record testing cost of the new model
        node.model.parameters = node.Theta[node.name]
        node.costs.append(node.model.test_cost(node.X_test, node.y_test))
        node.stop_condition -= 1
    # else:
    #     print(f"{node.pname} >> SC reached {node.stop_condition}.")


def cl_update_secondary(node, data):
    """update secondary variables."""
    mjj = data['payload']['model_i']
    mji = data['payload']['model_j']
    Ajj = data['payload']['A_i']
    Aji = data['payload']['A_j']
    i = node.name
    j = data['sender']['name']
    node.Z[i] = 1 / 2 * (1 / node.rho * (node.A[i] + Aji + node.Theta[i] + mji))
    node.Z[j] = 1 / 2 * (1 / node.rho * (Ajj + node.A[j] + mjj + node.Theta[j]))


def cl_update_dual(node, data):
    """update dual variables."""
    Ajj = data['payload']['A_i']
    Aji = data['payload']['A_j']
    i = node.name
    j = data['sender']['name']
    node.A[i] = node.A[i] + node.rho * (node.Theta[i] - node.Z[i])
    node.A[j] = node.A[j] + node.rho * (node.Theta[j] - node.Z[j])
