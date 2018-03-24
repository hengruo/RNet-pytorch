import argparse

def parse_args():
    args = argparse.ArgumentParser(description="An R-net implementation.")
    args.add_argument('--mode', dest='mode', type=str, default='all')
    args.add_argument("--batch_size", dest='batch_size', type=int, default="64")
    args.add_argument("--checkpoint", dest='checkpoint', type=int, default="1000")
    args.add_argument("--num_steps", dest='num_steps', type=int, default="60000")
    return args.parse_args()

def main():
    args = parse_args()
    if args.mode == 'all':
        pass
    else:
        pass

if __name__ == '__main__':
    main()