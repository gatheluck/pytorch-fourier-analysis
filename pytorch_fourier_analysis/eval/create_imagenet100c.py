import tqdm
import shutil
import pathlib
from typing import Set


def main(
    imagenet100_rootpath: pathlib.Path,
    imagenet_c_rootpath: pathlib.Path,
    imagenet100_c_rootpath: pathlib.Path,
):
    imagenet100_wordnet_ids = _get_imagenet100_wordnet_ids(
        pathlib.Path(imagenet100_rootpath)
    )

    imagenet_c_rootpath = pathlib.Path(imagenet_c_rootpath)
    imagenet100_c_rootpath = pathlib.Path(imagenet100_c_rootpath)
    corruptions = _get_corruptions(imagenet_c_rootpath)
    with tqdm.tqdm(total=len(corruptions), ncols=80) as pbar:
        for corruption in corruptions:
            for level in [str(i) for i in range(1, 6)]:
                for wordnet_id in imagenet100_wordnet_ids:
                    source_dir = imagenet_c_rootpath / corruption / level / wordnet_id
                    target_dir = (
                        imagenet100_c_rootpath / corruption / level / wordnet_id
                    )
                    if not source_dir.exists():
                        raise ValueError("path {} does not exist".format(source_dir))
                    _copy_images(source_dir, target_dir)
            pbar.update()


def _get_imagenet100_wordnet_ids(imagenet100_rootpath: pathlib.Path) -> Set[str]:
    """
    Args
        imagenet100_rootpath: path to imagenet100. eg.) /media/datasets/imagenet100
    """
    imagenet100_valpath = imagenet100_rootpath / "val"
    if not imagenet100_valpath.exists():
        raise ValueError("path {} does not exist".format(imagenet100_valpath))

    return {p.name for p in imagenet100_valpath.iterdir() if p.is_dir()}


def _get_corruptions(imagenet_c_rootpath: pathlib.Path) -> Set[str]:
    """
    Args
        imagenet_c_rootpath: path to imagenet-c. eg.) /media/datasets/imagenet-c
    """
    if not imagenet_c_rootpath.exists():
        raise ValueError("path {} does not exist".format(imagenet_c_rootpath))

    return {p.name for p in imagenet_c_rootpath.iterdir() if p.is_dir()}


def _copy_images(source_dir: pathlib.Path, target_dir: pathlib.Path) -> None:
    """
    Args
        source_dir: path to source dir of copy. eg.) /media/datasets/imagenet-c/brightness/1
        target_dir: path to target dir to copy. eg.) /media/datasets/imagenet100-c/brightness/1
    """
    target_dir.mkdir(parents=True)

    for source_filepath in source_dir.iterdir():
        if source_filepath.suffix not in {".JPEG", ".png"}:
            continue

        target_filepath = target_dir / source_filepath.name
        # print("file {source_filepath} copy as {target_filepath}".format(source_filepath=source_filepath, target_filepath=target_filepath))
        shutil.copy2(source_filepath, target_filepath)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--imagenet100_rootpath", type=str, help="root path to ImageNet100"
    )
    parser.add_argument(
        "--imagenet_c_rootpath", type=str, help="root path to ImageNet-C"
    )
    parser.add_argument(
        "--imagenet100_c_rootpath", type=str, help="root path to ImageNet100-C"
    )
    opt = vars(parser.parse_args())

    opt["imagenet100_rootpath"] = pathlib.Path(opt["imagenet100_rootpath"])
    opt["imagenet_c_rootpath"] = pathlib.Path(opt["imagenet_c_rootpath"])
    opt["imagenet100_c_rootpath"] = pathlib.Path(opt["imagenet100_c_rootpath"])
    main(**opt)
