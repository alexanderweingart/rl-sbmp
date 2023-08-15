from artist.AckermannCarArtist import AckermannCarArtist
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("map_path")
    command_parsers = parser.add_subparsers(dest="command")
    fixed_start = command_parsers.add_parser("fixed_start")
    fixed_start.add_argument("x", type=float)
    fixed_start.add_argument("y", type=float)
    fixed_start.add_argument("theta", type=float)
    args = parser.parse_args()
    artist = AckermannCarArtist(0.25, "human", 10, map_path=args.map_path)
    if args.command == "fixed_start":
        artist.collision_checker_test(args.map_path, (args.x, args.y, args.theta))
    else:
        artist.collision_checker_test(args.map_path)

