import argparse
import sys
import time

from table_detection import TableDetector
from controller import GameController
from display import Display
from player import Player

parser = argparse.ArgumentParser()
parser.add_argument('--color')
parser.add_argument('--repeat', action='store_true')    
args = parser.parse_args(sys.argv[1:])

while True:
    # Get preliminary screenshot.
    prelim_screenshot = GameController.get_screen_image()
    prelim_td = TableDetector()
    prelim_td.load_image(prelim_screenshot)
    prelim_td.detect_all()

    # Determine where to place the cue for real screenshot.
    prelim_controller = GameController(
        prelim_td.tableSize, prelim_td.tableCropTopLeft, prelim_td.balls, prelim_td.ballRadius)
    stick_target = prelim_controller.find_stick_position()
    prelim_controller.move_mouse(stick_target)

    # Get real screenshot.
    screenshot = GameController.get_screen_image()
    td = TableDetector()
    td.load_image(screenshot)
    td.detect_all()
    td.remove_nondup_balls(prelim_td.balls)
    detections_img = screenshot.split('.')[0] + '-detections.png'
    td.save_table_detections(detections_img)
    print(f"Detections: {detections_img}")

    player = Player(td.tableSize, td.pockets, td.balls, td.ballRadius, args.color)
    target, force = player.get_shot()
    print(f"Aiming at {target} with force {force}.")

    controller = GameController(td.tableSize, td.tableCropTopLeft, td.balls, td.ballRadius)
    controller.make_shot(target, force)

    if args.repeat:
        time.sleep(10)
    else:
        break
