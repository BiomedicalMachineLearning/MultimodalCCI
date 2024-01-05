def normalise_samples_to_target(samples, target=None):
    """Normalizes LR matrices within a list of samples to a target sample based on 
    number of spots to account for differences in sample size.

    Args:
        samples: A list of anndata samples that have been run through stLearn CCI
        target (optional): A target sample to normalize to. If not provided, the first 
        sample's shape is used.

    Returns:
        The list of samples with normalized matrices.
    """

    if target is None:
        target = samples[0].shape[0]

    for i, sample in enumerate(samples):
        for lr, matrix in sample.uns["per_lr_cci_cell_type"].items():
            samples[i].uns["per_lr_cci_cell_type"][lr] = matrix * (
                target.shape[0] / sample.shape[0]
            )

    return samples


def get_majority_lr_pairs(samples, equal_to=False):
    """Identifies the LR pairs present in a majority of samples.

    Args:
        samples: A list of anndata samples that have been run through stLearn CCI
        equal_to (optional): If True, includes LR pairs present in exactly half of the 
        samples. Defaults to False.

    Returns:
        A list of LR pairs that are present in a majority of samples.
    """

    lr_pairs_counts = {}
    lr_pairs = []

    for sample in samples:
        for lr_pair, matrix in sample.uns["per_lr_cci_cell_type"].items():
            if sum(sum(matrix.values)) != 0:
                lr_pairs_counts[lr_pair] = lr_pairs_counts.setdefault(lr_pair, 0) + 1

    for lr_pair, count in lr_pairs_counts.items():
        if equal_to:
            if count >= len(samples) / 2:
                lr_pairs.append(lr_pair)
        else:
            if count > len(samples) / 2:
                lr_pairs.append(lr_pair)

    return lr_pairs


def get_avg_lr_pairs(samples, lr_pairs):
    """Calculates the average matrix for each LR pair across a list of samples.

    Args:
        samples: A list of anndata samples that have been run through stLearn CCI
        lr_pairs: A list of LR pairs to calculate averages for.

    Returns:
        A dictionary where keys are LR pairs and values are the corresponding average 
        matrices.
    """

    lr_pairs_matrices = {}

    for sample in samples:
        for lr_pair, matrix in sample.uns["per_lr_cci_cell_type"].items():

            if lr_pair in lr_pairs:
                current_matrix = lr_pairs_matrices.get(lr_pair)

                if current_matrix is not None:
                    common_rows = list(set(matrix.index) & set(current_matrix.index))
                    common_cols = list(
                        set(matrix.columns) & set(current_matrix.columns)
                    )
                    matrix = matrix.loc[common_rows, common_cols]
                    current_matrix = current_matrix.loc[common_rows, common_cols]
                    lr_pairs_matrices[lr_pair] = current_matrix + matrix / len(
                        samples
                    )
                else:
                    lr_pairs_matrices[lr_pair] = matrix / len(samples)

    return lr_pairs_matrices


def normalise_samples_between_tech(samples, method="mean", tech_norm=0):
    """Normalizes matrices between different technologies.

    Args:
        samples: A nested list of samples, organized by technology and then group. 
        Example: [[t1g1, t1g2], [t2g1, t2g2]].
        method (optional): The normalization method, either "mean" or "sum". Defaults to
        "mean". Normalises based on either the mean or sum of the values in the matrix.
        tech_norm (optional): The index of the technology to use as the normalization 
        reference. Defaults to 0.

    Returns:
        The normalized nested list of samples.
    """

    tech_counts = []
    for tech_samples in samples:

        group_counts = []
        for group_samples in tech_samples:

            total_counts = 0
            for pair, matrix in group_samples.items():
                if method == "mean":
                    total_counts += matrix.mean().mean()
                elif method == "sum":
                    total_counts += matrix.sum().sum()
                else:
                    raise ValueError("Invalid method option.")

            group_counts.append(total_counts)
        tech_counts.append(statistics.mean(group_counts))

    for tech in range(len(samples)):
        for group in range(len(samples[tech])):
            for lr, df in samples[tech][group].items():
                factor = tech_counts[tech_norm] / tech_counts[tech]
                samples[tech][group][lr] = df * factor

    return samples


def integrate_between_tech(samples, method="multiply"):
    """Integrates matrices from different technologies using a specified method.

    Args:
        samples: A list of samples with different technologies.
        method (optional): The integration method, currently supports "multiply". 
        Defaults to "multiply".

    Returns:
        A dictionary where keys are LR pairs and values are the integrated matrices.
    """

    integrated = {}

    if method == "multiply":
        lr_pairs = None
        for tech in range(len(samples)):
            if lr_pairs is None:
                lr_pairs = set(samples[tech].keys())
            else:
                lr_pairs = lr_pairs.intersection(set(samples[tech].keys()))
        for lr in sorted(lr_pairs):
            matrix = None
            for tech in range(len(samples)):
                if matrix is None:
                    matrix = samples[tech][lr]
                    matrix = matrix.fillna(0)
                else:
                    matrix = matrix * samples[tech][lr]
                    matrix = matrix.fillna(0)
            integrated[lr] = matrix
    else:
        raise ValueError("Invalid method option.")

    return integrated


def calculate_overall_interactions(sample):
    """Calculates an overall interaction matrix by combining matrices for different LR 
    pairs within a sample.

    Args:
        sample: A sample containing matrices for different LR pairs.

    Returns:
        A matrix representing the overall interactions.
    """

    total = None
    for lr in sample.keys():
        if sample[lr].sum().sum() > 0:
            if total is not None:
                total = total + sample[lr] / sample[lr].sum().sum()
                total = total.fillna(0)
            else:
                total = sample[lr] / sample[lr].sum().sum()
                total = total.fillna(0)

    if total is None:
        print("Integration Failed: No overlap between technologies")

    total = total / total.sum().sum()
    total = total.fillna(0)

    return total


def calculate_dissim(sample1, sample2):
    """Calculates a dissimilarity score between two samples for each common LR pair.

    Args:
        sample1, sample2: Two samples containing matrices for LR pairs.

    Returns:
        A dictionary where keys are common LR pairs and values are the dissimilarity 
        scores.
    """

    dissims = {}
    for lr in set(sample1.keys()).intersection(set(sample2.keys())):
        dissims[lr] = dissimilarity_score(sample1[lr], sample2[lr])

    return dissims


def get_lrs_per_celltype(samples, sender, reciever):
    """Compares LR pairs between two samples for specific cell types.

    Args:
        samples (list): A list of dictionaries containing LR matrices for the
        first sample.
        sender (str): The sender cell type.
        reciever (str): The receiver cell type.

    Returns:
        dict: A dictionary with keys 'sample1', 'sample2', etc., each containing
        a set of LR pairs and proportion of its weighting.
    """

    names = []

    samples = [aged_integrated, young_integrated]
    sender = "Vascular"
    reciever = "Astrocytes"

    for i in range(len(samples)):
        samples[i] = {
            key: df.loc[[sender]]
            for key, df in samples[i].items()
            if sender in df.index
        }

        samples[i] = {
            key: df[[reciever]]
            for key, df in samples[i].items()
            if reciever in df.columns
        }

        samples[i] = {
            key: df
            for key, df in samples[i].items()
            if not df.map(lambda x: x == 0).all().all()
        }

        samples[i] = {
            key: df
            for key, df in samples[i].items()
            if not df.map(lambda x: x == 0).all().all()
        }
        names.append("sample" + str(i + 1))

    data = {}

    for group_name, lr_dict in zip(names, samples):

        lr_props = {}
        total = 0
        for lr_pair in set(lr_dict.keys()):
            score = lr_dict[lr_pair].at[sender, reciever]
            total += score

        for lr_pair in set(lr_dict.keys()):
            lr_props[lr_pair] = lr_dict[lr_pair].at[sender, reciever] / total

        lr_props = dict(
            sorted(lr_props.items(), key=lambda item: item[1], reverse=True)
        )

        data[group_name] = lr_props

    return data
