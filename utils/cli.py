import argparse

def get_main_parser():
    """
    Create the main argument parser with all subcommands.
    
    Returns:
        The configured main parser
    """
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis with Attention Visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    return parser

def parse_args(args=None):
    """
    Parse command line arguments.

    Args:
        args: Command line arguments (defaults to sys.argv[1:])
    
    Returns:
        Parsed arguments
    """
    parser = get_main_parser()
    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help()
        exit(1)

    return parsed_args