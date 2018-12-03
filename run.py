import argparse
import sys
import time

from table_detection import TableDetector
from controller import GameController
from display import Display
from player import Player

parser = argparse.ArgumentParser()
parser.add_argument('--color')
parser.add_argument('--flex', action='store_true')
parser.add_argument('--repeat', action='store_true')
args = parser.parse_args(sys.argv[1:])

while True:
    # Get preliminary screenshot.
    print("Taking preliminary screenshot...")
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
    print("Taking real screenshot...")
    screenshot = GameController.get_screen_image()
    td = TableDetector()
    td.load_image(screenshot)
    td.detect_all()
    td.remove_nondup_balls(prelim_td.balls)
    detections_img = screenshot.split('.')[0] + '-detections.png'
    td.save_table_detections(detections_img)

    player = Player(td.tableSize, td.pockets, td.balls, td.ballRadius, args.color)
    shot = player.get_shot(flex=args.flex)
    if shot.is_break:
        print("Breaking...")
    if shot.second_target is not None:
        print("Rebound shot...")
    if shot.is_hail_mary:
        print("Hail mary...")
    print(f"target: {shot.target}, force: {shot.force()}, quality: {shot.quality()}")

    controller = GameController(td.tableSize, td.tableCropTopLeft, td.balls, td.ballRadius)
    # controller.make_shot(shot.target, shot.force())

    if args.repeat:
        time.sleep(10)
    else:
        break
