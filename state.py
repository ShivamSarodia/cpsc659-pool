class Boundary:
    def __init__(self, p1, p2):
        # List containing the two endpoints of the boundary.
        # Each endpoint is an (x,y) tuple.
        self.ends = [p1, p2]

class PoolState:
    def __init__(self):
        # List of coordinate pair tuples for center of each pocket.
        self.pockets = []

        # List of Boundary objects for the table boundaries.
        self.boundaries = []

        # The radius in pixels of each ball
        self.ball_radius = None

        # Position of the white ball
        self.white_pos = None

        # Position of the eight ball
        self.eight_pos = None

        # List of positions of the solid balls
        self.solids = []

        # List of positions of the stripe balls
        self.stripes = []

        # Current ball to hit in. One of:
        #    "SOLID"
        #    "STRIPE"
        #    "BLACK"
        self.ball_type = None