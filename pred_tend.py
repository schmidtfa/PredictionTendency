import numpy as np

def gen_fwd_mask(y):
    fwds = [False]

    max_val, min_val = np.max(y), np.min(y)

    for idx, cur_y in enumerate(y):
        if idx > 0:
            if y[idx-1] < cur_y:
                fwds.append(True)
            elif np.logical_and(cur_y == min_val, y[idx-1] == max_val):
                fwds.append(True)
            else:
                fwds.append(False)

    return fwds

def gen_rep_mask(y):
    reps = [False]

    for idx, cur_y in enumerate(y):
        if idx > 0:
            if y[idx-1] == cur_y:
                reps.append(True)
            else:
                reps.append(False)

    return reps


def extract_fwd_reps(X, y):

    fwd_mask = gen_fwd_mask(y)
    rep_mask = gen_rep_mask(y)

    X_fwd, y_fwd = X[fwd_mask], y[fwd_mask]
    X_rep, y_rep = X[rep_mask], y[rep_mask]

    return X_fwd, y_fwd, X_rep, y_rep


def _compute_dvals(clf, probas, y_rep):

    # extract probabilities for fwd transitions
    fwd_dict = dict(zip(clf.classes_, [1,2,3,0]))
    rep_dict = dict(zip(clf.classes_, [0,1,2,3]))

    fwd_indices = np.array([fwd_dict[key] for key in y_rep])
    rep_indices = np.array([rep_dict[key] for key in y_rep])

    fwd_tend, rep_tend = [], []
    for idx, (fwd, rep) in enumerate(zip(fwd_indices, rep_indices)):

        fwd_tend.append(probas[idx,:,fwd])
        rep_tend.append(probas[idx,:,rep])

    #extract
    dvals = (np.array(fwd_tend) / (np.array(fwd_tend) + np.array(rep_tend))).mean(axis=0)

    dvals_zero_centered = dvals - 0.5

    return dvals_zero_centered


def compute_dvals(clf, X_or, y_or, X_rd, y_rd):

    # transform labels
    or_dict = dict(zip(np.unique(y_or), [3,4,5,6]))
    rd_dict = dict(zip(np.unique(y_rd), [3,4,5,6]))

    y_or_new = np.array([or_dict[key] for key in y_or])
    y_rd_new = np.array([rd_dict[key] for key in y_rd])

    #get transitions and repetitions
    X_fwd_or, y_fwd_or, X_rep_or, y_rep_or = extract_fwd_reps(X_or, y_or_new)
    _, _, X_rep_rd, y_rep_rd = extract_fwd_reps(X_rd, y_rd_new)

    #fit the classifier on fwd transitions
    clf.fit(X_fwd_or, y_fwd_or)
    #predict on repetitions
    probas_or = clf.predict_proba(X_rep_or)
    probas_rd = clf.predict_proba(X_rep_rd)
    #get dvals
    dvals_or = _compute_dvals(clf, probas_or, y_rep_or)
    dvals_rd = _compute_dvals(clf, probas_rd, y_rep_rd)

    return dvals_or, dvals_rd


def get_pred_tend(dvals_or, dvals_rd, time_mask):

    rd_pre = dvals_rd[time_mask].copy()
    or_pre = dvals_or[time_mask].copy()

    or_pre[or_pre < 0] = 0
    rd_pre[rd_pre < 0] = 0

    pred_tend = (or_pre - rd_pre).sum()

    return pred_tend