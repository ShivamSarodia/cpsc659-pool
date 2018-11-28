import autopy
import random
import time

from table_detection import TableDetector

time.sleep(2)
screenshot = autopy.bitmap.capture_screen()
screenshot_name = "screenshots/screen_" + str(random.randint(0, 1e10)) + ".png"
screenshot.save(screenshot_name)

td = TableDetector()
td.load_image(screenshot_name)
td.detect_all()

td.display_table_detections()
td.produce_classification_data()