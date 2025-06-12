import argparse
import pickle
from pathlib import Path

import numpy as np


def compute_success_rate(trans_errors, rot_errors, translation_threshold,
                         rotation_threshold) -> float:
    successful_translation = np.array(trans_errors) < translation_threshold
    successful_rotation = np.array(rot_errors) < rotation_threshold
    success_rate = np.mean(successful_translation & successful_rotation)
    return success_rate


def main(file: Path):
    with open(file, "rb") as f:
        # pickle.dump({"rot": node.rot_errors, "trans": node.trans_errors}, f)
        data = pickle.load(f)

    rot_errors = data["rot"]
    trans_errors = data["trans"]
    for k, v in rot_errors.items():
        rot_errors[k] = np.array(v)
    for k, v in trans_errors.items():
        trans_errors[k] = np.array(v)

    # Filter successful transformations
    success = {}
    for method, rot_error in rot_errors.items():
        trans_error = trans_errors[method]
        success[method] = np.logical_and(trans_error < .6, rot_error < 1.5)

    error_string = ""
    for method, rot_error in rot_errors.items():
        if 'icp' in method and 'vfm' not in method:
            continue
        trans_error = trans_errors[method]
        recall = success[method]
        error_string += f"{method}\t{np.round(np.mean(trans_error), 2):.2f}$\pm${np.round(np.std(trans_error), 2):.2f}"
        error_string += f" & {np.round(np.mean(rot_error),2):.2f}$\pm${np.round(np.std(rot_error),2):.2f}"
        error_string += f" & {np.round(np.mean(recall)*100,2):.2f}"
        try:
            recall = success[f"{method}_icp"]
        except:
            pass
        error_string += f" & {np.round(np.mean(recall)*100,2):.2f}"
        error_string += "\n"
    with open(Path(__file__).parent / 'error.txt', "w") as f:
        f.write(error_string)
    print('=' * 80)
    for method, rot_error in rot_errors.items():
        str = f"Rotation error ({method:<20}): {np.mean(rot_error):.3f} ± {np.std(rot_error):.3f}"
        str = f"{str:<57}" + f" | {np.median(rot_error):.3f}"
        if success[method].any():
            str = f"{str:<67}" + f" | {rot_error[success[method]].mean():.3f} ± {rot_error[success[method]].std():.3f}"
        print(str)
    print('-' * 80)
    for method, trans_error in trans_errors.items():
        str = f"Translat error ({method:<20}): {np.mean(trans_error):.3f} ± {np.std(trans_error):.3f}"
        str = f"{str:<57}" + f" | {np.median(trans_error):.3f}"
        if success[method].any():
            str = f"{str:<67}" + f" | {trans_error[success[method]].mean():.3f} ± {trans_error[success[method]].std():.3f}"
        print(str)
    print('-' * 80)
    thresholds = [
        (.3, 15),  # PointDSC
        (.6, 1.5),  # GCL
        (2, 5),  # D3Feat, SpinNet
    ]
    str = f"{'':<20}: "
    for threshold in thresholds:
        str += f"{threshold[0]:>3}, {threshold[1]:<3} | "
    print(str[:-2])
    for method, rot_error in rot_errors.items():
        trans_error = trans_errors[method]
        str = f"{method:<20}: "
        for threshold in thresholds:
            str += f"{100 * compute_success_rate(trans_error, rot_error, *threshold):>8.2f} | "
        print(str[:-2])
    print('=' * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    args = parser.parse_args()

    file = Path(args.file)
    if file.suffix == ".pkl":
        main(file)

    elif file.is_dir():
        files = sorted(list(file.iterdir()))
        for file in files:
            print(f"File: {file.name}")
            main(file)

    else:
        raise ValueError("Invalid file")
