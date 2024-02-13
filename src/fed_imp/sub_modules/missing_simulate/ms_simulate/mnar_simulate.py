import numpy as np
import random
from sklearn.feature_selection import mutual_info_regression
from scipy.special import expit
from scipy import optimize


########################################################################################################################
# Self Censoring Sigmoid
########################################################################################################################
def simulate_nan_mnar_sigmoid(
        data, cols, missing_ratios, missing_funcs, strict=False, corr_type='self', seed=1002031
):
    mask = np.zeros(data.shape, dtype=bool)

    # add missing for each column
    for col in cols:

        if isinstance(missing_ratios, dict):
            missing_ratio = missing_ratios[col]
        elif isinstance(missing_ratios, list):
            missing_ratio = missing_ratios[cols.index(col)]
        else:
            missing_ratio = missing_ratios

        if isinstance(missing_funcs, dict):
            missing_func = missing_funcs[col]
        elif isinstance(missing_funcs, list):
            missing_func = missing_funcs[cols.index(col)]
        else:
            missing_func = missing_funcs

        # set the seed
        seed = (seed + 1203941) % (2 ^ 32 - 1)

        # missing is associated with column itself
        if corr_type == 'self':
            data_corr = data[:, col]
        elif corr_type == 'others':
            data_corr = data[:, [i for i in cols if i != col]]
        elif corr_type == 'all':
            data_corr = data
        else:
            raise NotImplementedError

        #################################################################################
        # pick coefficients and mask missing values
        #################################################################################
        mask = mask_mar_sigmoid(mask, col, data_corr, missing_ratio, missing_func, strict, seed)

    # assign the missing values
    data_ms = data.copy()
    data_ms[mask] = np.nan

    return data_ms


########################################################################################################################
# Self Censoring Quantile
########################################################################################################################
def simulate_nan_mnar_quantile(
        data, cols, missing_ratios, missing_funcs='left', strict=True, seed=201030
):
    # find the columns that are not to be adding missing values
    mask = np.zeros(data.shape, dtype=bool)

    for col in cols:

        if isinstance(missing_ratios, dict):
            missing_ratio = missing_ratios[col]
        elif isinstance(missing_ratios, list):
            missing_ratio = missing_ratios[cols.index(col)]
        else:
            missing_ratio = missing_ratios

        if isinstance(missing_funcs, dict):
            missing_func = missing_funcs[col]
        elif isinstance(missing_funcs, list):
            missing_func = missing_funcs[cols.index(col)]
        else:
            missing_func = missing_funcs

        # set the seed
        seed = (seed + 10087651) % (2 ** 32 - 1)
        random.seed(seed)
        np.random.seed(seed)

        data_corr = data[:, col]

        # find the quantile of the most correlated column
        if missing_func == 'random':
            missing_func = random.choice(['left', 'right', 'mid', 'tail'])

        # get mask based on quantile
        mask = mask_mar_quantile(mask, col, data_corr, missing_ratio, missing_func, strict, seed)

    # assign the missing values
    data_ms = data.copy()
    data_ms[mask] = np.nan

    return data_ms


########################################################################################################################
# Helper Functions
########################################################################################################################
def mask_mar_sigmoid(mask, col, data_corr, missing_ratio, missing_func, strict, seed):
    np.random.seed(seed)
    random.seed(seed)

    #################################################################################
    # pick coefficients
    #################################################################################
    # Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    if isinstance(missing_ratio, dict):
        missing_ratio = missing_ratio[col]
    else:
        missing_ratio = missing_ratio

    # copy data and do min-max normalization
    data_copy = data_corr.copy()

    if data_copy.ndim == 1:
        data_copy = data_copy.reshape(-1, 1)

    data_copy = (data_copy - data_copy.min(0, keepdims=True)) / (
            data_copy.max(0, keepdims=True) - data_copy.min(0, keepdims=True) + 1e-5)
    data_copy = (data_copy - data_copy.mean(0, keepdims=True)) / \
                (data_copy.std(0, keepdims=True) + 1e-5)

    coeffs = np.random.rand(data_copy.shape[1], 1)
    Wx = data_copy @ coeffs
    wss = (Wx) / (np.std(Wx, 0, keepdims=True) + 1e-3)

    def f(x: np.ndarray) -> np.ndarray:
        if missing_func == 'left':
            return expit(-wss + x).mean().item() - missing_ratio
        elif missing_func == 'right':
            return expit(wss + x).mean().item() - missing_ratio
        elif missing_func == 'mid':
            return expit(np.absolute(wss) - 0.75 + x).mean().item() - missing_ratio
        elif missing_func == 'tail':
            return expit(-np.absolute(wss) + 0.75 + x).mean().item() - missing_ratio
        else:
            raise NotImplementedError

    intercept = optimize.bisect(f, -50, 50)

    if missing_func == 'left':
        ps = expit(-wss + intercept)
    elif missing_func == 'right':
        ps = expit(wss + intercept)
    elif missing_func == 'mid':
        ps = expit(-np.absolute(wss) + 0.75 + intercept)
    elif missing_func == 'tail':
        ps = expit(np.absolute(wss) - 0.75 + intercept)
    else:
        raise NotImplementedError

    # strict false means using random simulation
    if strict is False:
        ber = np.random.binomial(n=1, size=mask.shape[0], p=ps.flatten())
        mask[:, col] = ber
    # strict mode based on rank on calculated probability, strictly made missing
    else:
        ps = ps.flatten()
        # print(ps)
        end_value = np.sort(ps)[::-1][int(missing_ratio * data_copy.shape[0])]
        indices = np.where((ps - end_value) > 1e-3)[0]
        if len(indices) < int(missing_ratio * data_copy.shape[0]):
            end_indices = np.where(np.absolute(ps - end_value) <= 1e-3)[0]
            end_indices = np.random.choice(
                end_indices, int(missing_ratio * data_copy.shape[0]) - len(indices), replace=False
            )
            indices = np.concatenate((indices, end_indices))
        elif len(indices) > int(missing_ratio * data_copy.shape[0]):
            indices = np.random.choice(
                indices, int(
                    missing_ratio * data_copy.shape[0]
                ), replace=False
            )

        mask[indices, col] = True

    return mask


def mask_mar_quantile(mask, col, data_corr, missing_ratio, missing_func, strict, seed):
    if strict:
        total_missing = int(missing_ratio * data_corr.shape[0])
        sorted_values = np.sort(data_corr)
        if missing_func == 'left':
            q = sorted_values[int(missing_ratio * data_corr.shape[0]) - 1]
            indices = np.where(data_corr < q)[0]

            if len(indices) < total_missing:
                end_indices = np.where(
                    data_corr == q
                )[0]
                add_up_indices = np.random.choice(
                    end_indices, size=total_missing - len(indices), replace=False
                )
                na_indices = np.concatenate((indices, add_up_indices))
            elif len(indices) > total_missing:
                na_indices = np.random.choice(
                    indices, size=total_missing, replace=False
                )
            else:
                na_indices = indices
        elif missing_func == 'right':
            q = sorted_values[int((1 - missing_ratio) * data_corr.shape[0])]
            indices = np.where(data_corr > q)[0]
            if len(indices) < total_missing:
                start_indices = np.where(
                    data_corr == q
                )[0]
                add_up_indices = np.random.choice(
                    start_indices, size=total_missing - len(indices), replace=False
                )
                na_indices = np.concatenate((indices, add_up_indices))
            elif len(indices) > total_missing:
                na_indices = np.random.choice(
                    indices, size=total_missing, replace=False
                )
            else:
                na_indices = indices
        elif missing_func == 'mid':
            q0 = sorted_values[int((1 - missing_ratio) / 2 * data_corr.shape[0])]
            q1 = sorted_values[int((1 + missing_ratio) / 2 * data_corr.shape[0]) - 1]
            indices = np.where(
                (data_corr > q0) & (data_corr < q1)
            )[0]
            if len(indices) < total_missing:
                end_indices_q0 = np.where(data_corr == q0)[0]
                end_indices_q1 = np.where(data_corr == q1)[0]
                end_indices = np.union1d(
                    end_indices_q0, end_indices_q1
                )
                add_up_indices = np.random.choice(
                    end_indices, size=total_missing - len(indices), replace=False
                )
                na_indices = np.concatenate((indices, add_up_indices))
            elif len(indices) > total_missing:
                na_indices = np.random.choice(
                    indices, size=total_missing, replace=False
                )
            else:
                na_indices = indices
        elif missing_func == 'tail':
            q0 = sorted_values[int(missing_ratio / 2 * data_corr.shape[0])]
            q1 = sorted_values[int((1 - missing_ratio / 2) * data_corr.shape[0]) - 1]
            indices = np.where(
                (data_corr < q0) | (data_corr > q1)
            )[0]

            if len(indices) < total_missing:
                end_indices_q0 = np.where(data_corr == q0)[0]
                end_indices_q1 = np.where(data_corr == q1)[0]
                print(missing_func, q0, q1, len(indices), total_missing, len(data_corr), len(end_indices_q0),
                      len(end_indices_q1))
                end_indices = np.union1d(
                    end_indices_q0, end_indices_q1
                )
                add_up_indices = np.random.choice(
                    end_indices, size=total_missing - len(indices), replace=False
                )
                na_indices = np.concatenate((indices, add_up_indices))
            elif len(indices) > total_missing:
                na_indices = np.random.choice(
                    indices, size=total_missing, replace=False
                )
            else:
                na_indices = indices
        else:
            raise NotImplementedError
    else:
        if missing_func == 'left':
            q0 = 0
            q1 = 0.5 if missing_ratio <= 0.5 else missing_ratio
        elif missing_func == 'right':
            q0 = 0.5 if missing_ratio <= 0.5 else 1 - missing_ratio
            q1 = 1
        elif missing_func == 'mid' or missing_func == 'tail':
            q0 = 0.25 if missing_ratio <= 0.5 else 0.5 - missing_ratio / 2
            q1 = 0.75 if missing_ratio <= 0.5 else 0.5 + missing_ratio / 2
        else:
            raise NotImplementedError

        sorted_values = np.sort(data_corr)
        q0 = sorted_values[int(q0 * data_corr.shape[0])]
        q1 = sorted_values[int(q1 * data_corr.shape[0]) - 1]

        if missing_func != 'tail':
            indices = np.where(
                (data_corr >= q0) & (data_corr <= q1)
            )[0]
        else:
            indices = np.where(
                (data_corr <= q0) | (data_corr >= q1)
            )[0]
        np.random.seed(seed)
        na_indices = np.random.choice(
            indices, size=int(missing_ratio * data_corr.shape[0]), replace=False
        )

    mask[na_indices, col] = True

    return mask
