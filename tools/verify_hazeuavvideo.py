"""Check that a local HazeUAVvideo copy matches the paper's split."""

import argparse
import json
from pathlib import Path


BETAS = {'0.005', '0.01', '0.02'}
DATASETS = {'UAV123', 'UAVDT', 'VisDrone'}
EXPECTED = {'train': (125, 20466), 'test': (39, 5630)}


def check_clean_split(root, split):
    official_path = (Path(__file__).resolve().parents[1] / 'splits' /
                     f'meta_info_tree_GT_{split}.json')
    official = json.loads(official_path.read_text(encoding='utf-8'))
    expected = {}
    for record in official.values():
        expected.setdefault(record['gt_key'], record['frames'])
    gt_root = root / split / 'gt'
    actual = {
        f'{dataset.name}/{sequence.name}'
        for dataset in gt_root.iterdir() if dataset.is_dir()
        for sequence in dataset.iterdir() if sequence.is_dir()
    }
    if actual != set(expected):
        raise ValueError(f'{split}: clean source sequences differ from official split')
    for source, frames in expected.items():
        present = {p.name for p in (gt_root / source).iterdir() if p.is_file()}
        if present != set(frames):
            raise ValueError(f'{split}: clean frames differ in {source}')
    print(f'{split}: {len(expected)} official clean source sequences verified')


def check_split(root, split, structure_only=False):
    split_root = root / split
    tree_path = split_root / f'meta_info_tree_GT_{split}.json'
    index_path = split_root / f'meta_info_GT_{split}.txt'
    tree = json.loads(tree_path.read_text(encoding='utf-8'))
    official_path = Path(__file__).resolve().parents[1] / 'splits' / tree_path.name
    if official_path.is_file() and not structure_only:
        official = json.loads(official_path.read_text(encoding='utf-8'))
        if set(tree) != set(official):
            raise ValueError(f'{split}: clip keys differ from the official split')
        for key, record in tree.items():
            reference = official[key]
            if record.get('gt_key') != reference.get('gt_key') or (
                    record.get('frames') != reference.get('frames')):
                raise ValueError(f'{split}: frames differ from official split in {key}')
    index = {}
    for line in index_path.read_text(encoding='utf-8').splitlines():
        key, count = line.split()
        index[key] = int(count)
    if set(tree) != set(index):
        raise ValueError(f'{split}: tree and text metadata have different clips')

    sources = {}
    for key, record in tree.items():
        dataset, clip = key.split('/', 1)
        if dataset not in DATASETS:
            raise ValueError(f'{split}: unexpected source dataset {dataset}')
        beta = clip.rsplit('_', 1)[-1]
        if beta not in BETAS:
            raise ValueError(f'{split}: unexpected haze density in {key}')
        source = record.get('gt_key')
        frames = record.get('frames')
        if not source or not frames:
            raise ValueError(f'{split}: incomplete metadata for {key}')
        if len(frames) != index[key] or not frames:
            raise ValueError(f'{split}: inconsistent frame count in {key}')
        hazy_frame = split_root / 'hazy' / key / frames[0]
        gt_frame = split_root / 'gt' / source / frames[0]
        if not hazy_frame.is_file() or not gt_frame.is_file():
            raise ValueError(f'{split}: missing paired frame for {key}')
        entry = sources.setdefault(source, {})
        entry[beta] = len(frames)

    for source, counts in sources.items():
        if set(counts) != BETAS or len(set(counts.values())) != 1:
            raise ValueError(f'{split}: incomplete density triplet for {source}')
    expected_sources, expected_frames = EXPECTED[split]
    frames = sum(next(iter(counts.values())) for counts in sources.values())
    if len(sources) != expected_sources or frames != expected_frames:
        raise ValueError(
            f'{split}: expected {expected_sources} sources/{expected_frames} clean '
            f'frames, found {len(sources)}/{frames}')
    print(f'{split}: {len(sources)} sources, {len(tree)} hazy clips, '
          f'{frames} clean frames, {frames * len(BETAS)} hazy frames')
    return set(sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--clean-only', action='store_true')
    parser.add_argument('--structure-only', action='store_true',
                        help='Check pairs, density triplets, and split size without matching official clip names')
    args = parser.parse_args()
    if args.clean_only:
        for split in EXPECTED:
            check_clean_split(args.root, split)
        return
    train_sources = check_split(args.root, 'train', args.structure_only)
    test_sources = check_split(args.root, 'test', args.structure_only)
    overlap = train_sources & test_sources
    if overlap:
        raise ValueError(f'source-sequence leakage: {sorted(overlap)[:5]}')
    label = 'Generated split structure' if args.structure_only else 'Paper split'
    print(f'{label} verified: 164 sources, 492 hazy clips, '
          '26096 clean frames, 78288 hazy frames')


if __name__ == '__main__':
    main()
