"""Generate a candidate external-model manifest; never change the shipped defaults."""
import argparse
import json
from pathlib import Path


def main():
    from huggingface_hub import HfApi
    parser = argparse.ArgumentParser(description=__doc__)
    for side in ('source', 'target'):
        parser.add_argument('--' + side + '-repo', required=True)
        parser.add_argument('--' + side + '-file', required=True)
        parser.add_argument('--' + side + '-revision', default='main')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for side in ('source', 'target'):
        repo = getattr(args, side + '_repo')
        name = getattr(args, side + '_file')
        info = HfApi().model_info(repo, revision=getattr(args, side + '_revision'), files_metadata=True)
        entry = next(item for item in info.siblings if item.rfilename == name)
        if not entry.lfs or not entry.lfs.sha256:
            raise ValueError(f'{repo}/{name}: missing LFS SHA-256; inspect manually')
        result[side] = dict(repo_id=repo, revision=info.sha, filename=name,
                            bytes=entry.size, sha256=entry.lfs.sha256)
    # Exclusive creation avoids accidentally overwriting a qualified manifest.
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(f'Candidate written to {args.output}; GPU qualification is still required.')


if __name__ == '__main__':
    main()
