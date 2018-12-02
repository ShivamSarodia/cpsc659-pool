import autopy
import numpy as np
import random
import time

class GameController:
    def __init__(self, table_size, crop_offset, balls, ball_radius):
        # Size of the table image in pixels as (width, height)
        self.table_size = table_size

        # Coordinates of the top left of the table image, relative to the entire screen image.
        self.crop_offset = crop_offset

        # Coordinates of the cue ball, relative to the table image.
        self.cue_coords = balls["white"]

        # Coordinates of all balls other than the cue ball, relative to the table image.
        self.other_balls = balls["solids"] + balls["stripes"]
        if "black" in balls:
            self.other_balls.append(balls["black"])

        self.ball_radius = ball_radius

    @staticmethod
    def get_screen_image(dir="screenshots/"):
        """Return filename of a PNG containing current screen contents."""
        screenshot_name = dir + "/screenshot_" + str(random.randint(0, 1e10)) + ".png"

        screenshot = autopy.bitmap.capture_screen()
        screenshot.save(screenshot_name)
        return screenshot_name

    def drag_and_drop(self, start, end):
        start_x, start_y = start
        end_x, end_y = end

        adj_start = ((start_x + self.crop_offset[0]) / 2,
                     (start_y + self.crop_offset[1]) / 2)
        adj_end = ((end_x + self.crop_offset[0]) / 2,
                   (end_y + self.crop_offset[1]) / 2)

        autopy.mouse.move(*adj_start)
        time.sleep(1)
        autopy.mouse.click()
        autopy.mouse.toggle(None, True)
        autopy.mouse.smooth_move(*adj_end)
        autopy.mouse.toggle(None, False)
        time.sleep(0.5)

    def move_mouse(self, point):                   
        autopy.mouse.move(
            (point[0] + self.crop_offset[0]) / 2,
            (point[1] + self.crop_offset[1]) / 2)
        time.sleep(0.5)

    def _get_edge_intersections(self, target):
        slope = (target[1] - self.cue_coords[1]) / (target[0] - self.cue_coords[0])
        x_to_y = lambda x: self.cue_coords[1] + slope * (x - self.cue_coords[0])
        y_to_x = lambda y: self.cue_coords[0] + (y - self.cue_coords[1]) / slope

        # Start the drag from an edge of the table image.
        intersections = []

        y = x_to_y(0)
        if y >= 0 and y <= self.table_size[1]:
            intersections.append((0, y))

        y = x_to_y(self.table_size[0])
        if y >= 0 and y <= self.table_size[1]:
            intersections.append((self.table_size[0], y))

        x = y_to_x(0)
        if x >= 0 and x <= self.table_size[0]:
            intersections.append((x, 0))

        x = y_to_x(self.table_size[1])
        if x >= 0 and x <= self.table_size[0]:
            intersections.append((x, self.table_size[1]))

        return intersections
    
    def make_shot(self, target, force):
        """Make a shot which launches the cue ball to make its center pass through the target point.
        
        Force is a value roughly between 0 and 1 that controls the force of the hit.
        """
        intersections = self._get_edge_intersections(target)
        print(intersections)

        if target[0] > self.cue_coords[0]:
            start = max(intersections, key=lambda i: i[0])
        elif target[0] < self.cue_coords[0]:
            start = min(intersections, key=lambda i: i[0])
        elif target[1] > self.cue_coords[1]:
            start = max(intersections, key=lambda i: i[1])
        elif target[1] < self.cue_coords[1]:
            start = min(intersections, key=lambda i: i[1])
        else:
            raise Exception("Could not find suitable start for shot.")

        drag_length = force * 500
        x_dist = target[0] - self.cue_coords[0]
        y_dist = target[1] - self.cue_coords[1]
        tot_dist = np.sqrt(x_dist ** 2 + y_dist ** 2)
        
        x_diff = (x_dist / tot_dist) * drag_length
        y_diff = (y_dist / tot_dist) * drag_length
        end = start[0] - x_diff, start[1] - y_diff

        self.drag_and_drop(start, end)

    def find_stick_position(self, retry=200):
        """Find a queue position for which the cue and guide do not obscure any balls."""

        for _ in range(retry):
            r = np.random.random()
            if r < 0.25:
                target = (0, self.table_size[1] * np.random.random())
            elif r < 0.5:
                target = (self.table_size[0], self.table_size[1] * np.random.random())
            elif r < 0.75:
                target = (self.table_size[0] * np.random.random(), 0)
            else:
                target = (self.table_size[0] * np.random.random(), self.table_size[1])
            
            if self._test_stick_position(target):
                return target

        # If nothing was found, return a random target.
        print("Could not find a good stick position for image.")
        return target

    def _test_stick_position(self, target):
        """Test whether a given stick position would not obscure any balls."""

        cue = np.array(self.cue_coords)
        target = np.array(target)

        # Get rotation matrix
        delta = target - cue
        l = np.linalg.norm(delta)
        rotation = np.array([[delta[1] / l, -delta[0] / l], [delta[0] / l, delta[1] / l]])

        rot_start = rotation.dot(target)
        rot_end = rotation.dot(cue)

        for ball in self.other_balls:
            rot_ball = rotation.dot(np.array(ball))
            dist = np.abs(rot_ball[0] - rot_start[0])
            if dist < 2.1 * self.ball_radius:
                return False

        return True
        



        

