# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------

import argparse
import os
import uuid

import Thesis.backup_models.main_gan as trainer

def parse_args():
    trainer_parser = trainer.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for MAE pretrain", parents=[trainer_parser])
    parser.add_argument("--job_dir", type=str, default="", help="Output directory")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.job_dir == "":
        args.job_dir = "./output/submitit_runs/mae_" + str(uuid.uuid4())[:8]
        os.makedirs(args.job_dir, exist_ok=True)
    
    args.output_dir = args.job_dir

    trainer.main(args)

if __name__ == "__main__":
    main()