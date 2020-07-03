import argparse
import logging

import fmlwright.dataset_generators as data_gen

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s"
)
log = logging.getLogger(__name__)

MODULE_OPTIONS = [
    "text_to_gdf",
    "ground_plan",
    "floor_plan",
    "structure_plan",
    "complete_floorplan",
]

STEP_OPTIONS = [
    "generate_images",
    "generate_dataset",
]


def run_module_structure_plan(args):
    """Create the windows, doors and entrance for the generated area.

    Args:
        args (argparse): Arguments added to the python script through argparse.
    """
    step = args.step
    img_quality = 70
    if step == "generate_images":
        StructureGenerator = data_gen.ImageStructureGenerator(
            input_directory=args.input_directory,
            output_directory=args.output_directory,
            img_quality=img_quality,
        )
        StructureGenerator.generate_images()
    elif step == "generate_dataset":
        DatasetGenerator = data_gen.DatasetGenerator(
            input_directory=args.input_directory, output_directory=args.output_directory
        )
        DatasetGenerator.generate_dataset()


def run_module_floor_plan(args):
    """Create the datasets for floor plans.

    Args:
        args (argparse): Arguments added to the python script through argparse.
    """
    step = args.step
    img_quality = 70
    if step == "generate_images":
        FloorplanGenerator = data_gen.ImageFloorplanGenerator(
            input_directory=args.input_directory,
            output_directory=args.output_directory,
            img_quality=img_quality,
        )
        FloorplanGenerator.generate_images()
    elif step == "generate_dataset":
        DatasetGenerator = data_gen.DatasetGenerator(
            input_directory=args.input_directory, output_directory=args.output_directory
        )
        DatasetGenerator.generate_dataset()


def run_module_complete_floorplan(args):
    """Create the datasets for complete floor plans.

    Args:
        args (argparse): Arguments added to the python script through argparse.
    """
    step = args.step
    img_quality = 70
    if step == "generate_images":
        FloorplanGenerator = data_gen.ImageSingleStepGenerator(
            input_directory=args.input_directory,
            output_directory=args.output_directory,
            img_quality=img_quality,
        )
        FloorplanGenerator.generate_images()
    elif step == "generate_dataset":
        DatasetGenerator = data_gen.DatasetGenerator(
            input_directory=args.input_directory, output_directory=args.output_directory
        )
        DatasetGenerator.generate_dataset()


def run_module_ground_plan(args):
    """Create the outline of a building footprint based on the borders of the area.

    Args:
        args (argparse): Arguments added to the python script through argparse.
    """
    raise NotImplementedError


def run_module_text_to_gdf(args):
    """Create a geojson file from the large dataset of text files.

    Args:
        args (argparse): Arguments added to the python script through argparse.
    """
    GeoGenerator = data_gen.GeoDataGenerator(
        input_directory=args.input_directory,
        output_directory=args.output_directory,
    )
    GeoGenerator.generate_dataset()


def main(args):
    """Main function that serves as an entry point for the dataset generators.

    Args:
        args: Arguments added to the python script through argparse.
    """
    module = args.module

    if args.step not in STEP_OPTIONS:
        print(f"{args.step} is an unknown option. Your options are {STEP_OPTIONS}")
        exit()

    if module == "structure_plan":
        run_module_structure_plan(args)
    elif module == "floor_plan":
        run_module_floor_plan(args)
    elif module == "complete_floorplan":
        run_module_complete_floorplan(args)
    elif module == "ground_plan":
        run_module_ground_plan(args)
    elif module == "text_to_gdf":
        run_module_text_to_gdf(args)
    else:
        print(f"{module} is an unknown option. Your options are {MODULE_OPTIONS}.")
        exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate text to geodataframe dataset."
    )
    parser.add_argument(
        "--module",
        type=str,
        metavar="module",
        help="Define what part of the pipeline you would like to run.",
    )
    parser.add_argument(
        "--step",
        type=str,
        metavar="step",
        help="Define what step of the module you would like to run.",
    )
    parser.add_argument(
        "--input_directory", type=str, metavar="input_directory", help="input directory"
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        metavar="output_directory",
        help="output directory",
    )

    args = parser.parse_args()

    main(args)
