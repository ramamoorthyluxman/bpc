import argparse
import os
import shlex
import threading

from rocker.core import DockerImageGenerator
from rocker.core import get_rocker_version
from rocker.core import RockerExtensionManager
from rocker.core import OPERATIONS_NON_INTERACTIVE


def main():

    parser = argparse.ArgumentParser(
        description="The entry point for the Bin Picking Challenge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("test_image")
    parser.add_argument("dataset_directory")
    parser.add_argument(
        "-v", "--version", action="version", version="%(prog)s " + get_rocker_version()
    )
    parser.add_argument("--debug-inside", action="store_true")

    extension_manager = RockerExtensionManager()
    default_args = {"cuda": True, "network": "host"}
    extension_manager.extend_cli_parser(parser, default_args)

    args = parser.parse_args()
    args_dict = vars(args)

    # Confirm dataset directory is absolute
    args_dict["dataset_directory"] = os.path.abspath(args_dict["dataset_directory"])

    active_extensions = extension_manager.get_active_extensions(args_dict)
    print("Active extensions %s" % [e.get_name() for e in active_extensions])

    tester_args = {
        "network": "host",
        "extension_blacklist": {},
        "operating_mode": OPERATIONS_NON_INTERACTIVE,
        "env": [[f"BOP_PATH:/opt/ros/underlay/install/datasets"]],
        "console_output_file": "ibpc_test_output.log",
        "volume": [
            [f"{args_dict['dataset_directory']}:/opt/ros/underlay/install/datasets/lm"]
        ],
    }
    print("Buiding tester env")
    tester_extensions = extension_manager.get_active_extensions(tester_args)
    dig_tester = DockerImageGenerator(tester_extensions, tester_args, "ibpc:tester")

    exit_code = dig_tester.build(**tester_args)
    if exit_code != 0:
        print("Build of tester failed exiting")
        return exit_code

    zenoh_args = {
        "network": "host",
        "extension_blacklist": {},
        "console_output_file": "ibpc_zenoh_output.log",
        "operating_mode": OPERATIONS_NON_INTERACTIVE,
    }

    print("Buiding zenoh env")
    dig_zenoh = DockerImageGenerator(
        tester_extensions, tester_args, "eclipse/zenoh:1.1.1"
    )
    exit_code = dig_zenoh.build(**zenoh_args)
    if exit_code != 0:
        print("Build of zenoh failed exiting")
        return exit_code

    def run_instance(dig_instance, args):
        dig_instance.run(**args)

    tester_thread = threading.Thread(target=run_instance, args=(dig_zenoh, zenoh_args))
    tester_thread.start()

    # TODO Redirect stdout
    import time

    time.sleep(3)

    tester_thread = threading.Thread(
        target=run_instance, args=(dig_tester, tester_args)
    )
    tester_thread.start()
    # TODO Redirect stdout

    import time

    time.sleep(3)

    dig = DockerImageGenerator(active_extensions, args_dict, args_dict["test_image"])

    exit_code = dig.build(**vars(args))
    if exit_code != 0:
        print("Build failed exiting")
        return exit_code

    if args.debug_inside:
        args_dict["command"] = "bash"

    result = dig.run(**args_dict)
    # TODO clean up threads here
    return result
