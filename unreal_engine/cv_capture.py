import setup_path
import airsim

import pprint
import tempfile
import os
import time
import numpy as np

from airsim import ImageType
from PIL import Image, ImageDraw
import io

# Initialize PrettyPrinter for better formatted output
pp = pprint.PrettyPrinter(indent=4)


def setup_client():
    """
    Set up the AirSim client and confirm connection.
    """
    client = airsim.VehicleClient()
    client.confirmConnection()
    client.simSetCameraFov("3", 87)
    return client


def create_output_directories(base_dir, num_dirs):
    """
    Create directories for saving images.
    """
    for n in range(num_dirs):
        os.makedirs(os.path.join(base_dir, str(n)), exist_ok=True)


def reset_segmentation_ids(client, object_name_list):
    """
    Reset segmentation object IDs and set new IDs for the specified objects.
    """
    print("Resetting all object IDs for segmentation")
    client.simSetSegmentationObjectID("[\w]*", 0, True)
    time.sleep(1)

    for idx, obj_name in enumerate(object_name_list):
        print(f"Setting ID for: {obj_name}")
        obj_name_reg = r"[\w]*" + obj_name + r"[\w]*"
        found = client.simSetSegmentationObjectID(obj_name_reg, (idx + 1) % 256, True)
        print(f"{obj_name}: {found}")


def capture_images(client, tmp_dir, meshNames, num_iterations=12):
    """
    Capture and save images along with their annotations.
    """
    for x in range(num_iterations):
        # Set the vehicle's pose
        client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x + 90, -20, -30), airsim.to_quaternion(0, 0, 0)), True)
        time.sleep(0.1)

        client.simPause(is_paused=True)

        # Get images
        responses = client.simGetImages([
            airsim.ImageRequest("3", ImageType.Scene, False, False),
            airsim.ImageRequest("3", ImageType.Segmentation, False, False)
        ])

        # Get detections
        dets = client.simGetDetections("3", ImageType.Scene)
        print(f"Number of detections: {len(dets)}")

        save_images_and_annotations(responses, dets, tmp_dir, meshNames, x)

        client.simPause(is_paused=False)


def save_images_and_annotations(responses, dets, tmp_dir, meshNames, iteration):
    """
    Save captured images and their annotations.
    """
    image_file = os.path.normpath(os.path.join(tmp_dir, str(0), f"{iteration}_0"))

    for i, response in enumerate(responses):
        if response.pixels_as_float:
            print(f"Type {response.image_type}, size {len(response.image_data_float)}")
            airsim.write_pfm(image_file + '.pfm', airsim.get_pfm_array(response))
        else:
            img_rgb = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width,
                                                                                       3)
            image_type_str = "_orig.jpg" if response.image_type == ImageType.Scene else "_seg.jpg"
            airsim.write_png(image_file + image_type_str, img_rgb)

    annotate_and_save_image(image_file, dets, meshNames)


def annotate_and_save_image(image_file, dets, meshNames):
    """
    Annotate images with bounding boxes and save them.
    """
    im = Image.open(image_file + "_orig.jpg")
    listBoxes = []

    for box in dets:
        for mesh in meshNames:
            if mesh in box.name:
                listBoxes.append(box)
                draw = ImageDraw.Draw(im)
                draw.rectangle(
                    [(box.box2D.min.x_val, box.box2D.min.y_val),
                     (box.box2D.max.x_val, box.box2D.max.y_val)],
                    outline="red"
                )
                draw.text((box.box2D.min.x_val, box.box2D.min.y_val), box.name)

    im.save(image_file + "_annot.jpg")
    save_bounding_boxes(image_file, listBoxes)


def save_bounding_boxes(image_file, listBoxes):
    """
    Save bounding boxes in a text file.
    """
    with open(image_file + '.txt', 'w') as f:
        for box in listBoxes:
            x1 = round((box.box2D.min.x_val + box.box2D.max.x_val) / 2 / 1024, 3)
            y1 = round((box.box2D.min.y_val + box.box2D.max.y_val) / 2 / 1024, 3)
            x2 = round((box.box2D.max.x_val - box.box2D.min.x_val) / 1024, 3)
            y2 = round((box.box2D.max.y_val - box.box2D.min.y_val) / 1024, 3)
            f.write(f'1 {x1} {y1} {x2} {y2}\n')


def main():
    tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
    print(f"Saving images to {tmp_dir}")
    create_output_directories(tmp_dir, 4)

    client = setup_client()
    airsim.wait_key('Press any key to get camera parameters')

    for camera_id in range(2):
        camera_info = client.simGetCameraInfo(str(camera_id))
        print(f"CameraInfo {camera_id}:")
        pp.pprint(camera_info)

    airsim.wait_key('Press any key to get images')

    object_name_list = ["Character"]
    reset_segmentation_ids(client, object_name_list)

    client.simSetDetectionFilterRadius("3", ImageType.Scene, 80000)  # in [cm]
    client.simAddDetectionFilterMeshName("3", ImageType.Scene, "*")

    capture_images(client, tmp_dir, object_name_list)

    # Workaround for reset in CV mode
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)), True)


if __name__ == "__main__":
    main()
