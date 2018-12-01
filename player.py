import numpy as np

class Player:
    def __init__(self, table_size, pockets, balls, ball_radius):
        self.table_size = table_size
        self.pockets = pockets
        self.balls = balls
        self.ball_radius = ball_radius

        assert self.balls["white"] is not None
        self.all_balls = self.balls["stripes"] + self.balls["solids"]
        self.all_balls.append(self.balls["white"])
        if "black" in self.balls:
            self.all_balls.append(self.balls["black"])

    def _is_same_ball(self, b1, b2):
        return np.abs(b1[0] - b2[0]) + np.abs(b1[1] - b2[1]) < 1

    def _get_rotation(self, p1, p2):
        """Produce a rotation matrix such that the first coordinates of p1' and p2' are equal."""
        delta = p2 - p1
        l = np.linalg.norm(delta)
        return np.array([[delta[1] / l, -delta[0] / l], [delta[0] / l, delta[1] / l]])

    def _clear_distance(self, start, end, excepts=[]):
        """
        If a point travels from `start` to `end`, what is the minimum distance
        between the point and the center of another ball during its trajectory?
        
        Ignores the centers of balls listed in `excepts`.
        """        
        start = np.array(start)
        end = np.array(end)

        rotation = self._get_rotation(start, end)
        rot_start = rotation.dot(start)
        rot_end = rotation.dot(end)

        min_dist = 2 * (self.table_size[0] + self.table_size[1])
        for ball in self.all_balls:
            # Skip balls that are very close to a ball in the excepts list.
            to_skip = False
            for b2 in excepts:
                if self._is_same_ball(ball, b2):
                    to_skip = True
            if to_skip:
                continue

            rot_ball = rotation.dot(np.array(ball))
            if min(rot_start[1], rot_end[1]) <= rot_ball[1] <= max(rot_start[1], rot_end[1]):
                dist = np.abs(rot_ball[0] - rot_start[0])
                min_dist = min(dist, min_dist)

        return min_dist

    def _is_clear(self, start, end, excepts=[]):
        """
        Can a ball travel from `start` to `end` without striking another ball?
        
        Ignores the balls listed in `excepts`.
        """  
        return self._clear_distance(start, end, excepts) > 2.2 * self.ball_radius
    
    def _get_shots(self):
        shots = []
        for ball in self.balls["solids"]:
            for pocket_lab in self.pockets:
                shot = self._create_shot(ball, self.pockets[pocket_lab])
                if shot:
                    shots.append(shot)
        return shots

    def _create_shot(self, ball, pocket):
        """Create a shot to launch given ball into given pocket.

        If no such shot exists, return None.
        """
        pocket = np.array(pocket)
        ball = np.array(ball)
        cue = np.array(self.balls["white"])

        # Is path from ball to pocket clear?
        if not self._is_clear(ball, pocket, [cue, ball]):
            return None            
        
        # Compute target position for cue ball
        unit_towards_target = (ball - pocket) / np.linalg.norm(ball - pocket)
        target = ball + 2 * self.ball_radius * unit_towards_target

        # Compute quality of angle from cue ball to target
        v1 = cue - target
        v2 = pocket - target
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        if cos_angle > -0.3:
            return None

        # Is path from cue to target clear?
        if not self._is_clear(cue, target, [cue, ball]):
           return None

        return Shot(target, pocket, -cos_angle, np.linalg.norm(pocket - ball))

    def get_shot(self):
        if self._is_break():
            print("Breaking...")
            return self.table_size[0] / 2, 0

        shots = self._get_shots()
        shots.sort(key=lambda shot: -shot.cos_angle)
        return shots[0].target

    def _is_break(self):
        not_near = 0
        for b1 in self.all_balls:
            nearby = 0
            for b2 in self.all_balls:
                ball_dist = np.linalg.norm(np.array(b1) - np.array(b2))
                if ball_dist < 2.5 * self.ball_radius:
                    nearby += 1
            if nearby < 2:
                not_near += 1

        # TODO: once none detection is implemented, reduce to 2.
        return not_near < 3

class Shot:
    def __init__(self, target, pocket, cos_angle, travel_dist):
        self.target = target[0], target[1]
        self.pocket = pocket[0], pocket[1]
        self.cos_angle = cos_angle  # larger is better
        self.travel_dist = travel_dist