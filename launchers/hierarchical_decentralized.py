from launchers.parsers import build_parser
from launchers.decentralized import DecentralizedLauncher

def parse_args():
    parser = build_parser(auction=True, transformation='subpolicy')
    args = parser.parse_args()
    args.hrl = True
    return args

if __name__ == '__main__':
    launcher = DecentralizedLauncher()
    launcher.main(parse_args)