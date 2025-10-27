from collections import defaultdict
import pygame
import math
import neat
import glob # For finding checkpoint files
from neat import Checkpointer # For saving/loading population state
import os
import visualize
import argparse

# --- Game Setup ---
pygame.init()

# Screen dimensions (must match your track.png)
WIDTH, HEIGHT = 1200, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Python Drift Car")

# NEAT meta-data
RAY_COUNT = 9
RAY_LENGTH = 250
MAX_STEPS_PER_CAR = 4000
TILE_SIZE = 20
STALL_CHECK_INTERVAL = 120 # Check every 120 frames (2 seconds at 60 FPS)
MIN_DISPLACEMENT_PER_CHECK = 30 # Must move at least 30 pixels in that time


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
GREY = (100, 100, 100)
GREEN = (0, 200, 0) # For rays


# Game variables
clock = pygame.time.Clock()
FPS = 60
running = True
START_POS = (781, 507) 


# --- Load Assets ---
try:
    # Load the track image
    track = pygame.image.load('data/track.png').convert()
    if track.get_size() != (WIDTH, HEIGHT):
        print(f"Error: 'track.png' must be {WIDTH}x{HEIGHT} pixels.")
        running = False
except FileNotFoundError:
    print("Error: 'track.png' not found.")
    print("Please create a 1200x600 image named 'track.png' with a black track on a white background.")
    running = False

# -----------------------------------------------------------------
# --- FIXED FUNCTION: is_on_road ---
# -----------------------------------------------------------------
def is_on_road(x, y):
    """Checks if a coordinate is on the track (i.e., not white)."""
    w, h = track.get_size()
    if x < 0 or x >= w or y < 0 or y >= h:
        return False # Off the screen is not "on road"

    # Get the color and check its Red component (or any component)
    # If the pixel is dark (e.g., < 100), it's part of the track
    try:
        color = track.get_at((int(x), int(y)))
        return color[0] < 100 # Assumes track is black/dark
    except IndexError:
        return False # Failsafe for coordinates just off-edge

# --- Car Class ---
class Car:
    def __init__(self, start_x, start_y):
        # Create the car's visual representation (a simple rectangle)
        self.original_image = pygame.Surface((30, 15), pygame.SRCALPHA)
        self.original_image.fill(RED)
        # Add a "windshield" to show the front
        pygame.draw.rect(self.original_image, GREY, (20, 0, 10, 15))
        

        # --- Physics Variables ---
        self.pos = pygame.math.Vector2(start_x, start_y) # Position vector
        self.vel = pygame.math.Vector2(0, 0)             # Velocity vector
        self.angle = 0                                   # Car's orientation in degrees
        self.distance = 0.0

        self.image = self.original_image
        self.rect = self.image.get_rect(center=(start_x, start_y))
        self.steps = 0 
        self.alive = True
        self.visited_tiles = set()
        start_tile = (int(self.pos.x / TILE_SIZE), int(self.pos.y / TILE_SIZE))
        self.visited_tiles.add(start_tile)


        # --- Physics Constants (TUNE THESE!) ---
        self.max_speed = 8         # Maximum forward speed
        self.max_reverse_speed = -3 # Maximum reverse speed
        self.engine_power = 0.15   # Acceleration force
        self.brake_power = 0.3     # Braking/reversing force
        self.turn_rate = 3         # How fast the car's body rotates
        self.friction = 0.98       # Natural deceleration (1.0 = no friction, 0.0 = total stop)
        
        # This is the key to drifting!
        # 1.0 = perfect grip (no drift)
        # 0.0 = perfect ice (no grip)
        # Try values between 0.05 and 0.2 for a drifty feel
        self.traction = 0.08       
        
        self.last_check_step = 0
        self.last_check_pos = self.pos.copy()
        self.ray_endpoints = [] # Stores ray end-points for drawing

    def update(self, keys):
        """Update the car's physics based on user input"""
        if not self.alive:
            return
        
        # Get input
        acceleration_input = 0
        if keys[pygame.K_UP]:
            acceleration_input = 1
        if keys[pygame.K_DOWN]:
            acceleration_input = -1 # This will be for braking and reverse

        steering_input = 0
        if keys[pygame.K_LEFT]:
            steering_input = 1
        if keys[pygame.K_RIGHT]:
            steering_input = -1

        # --- 1. Steering ---
        # Only allow turning if the car is moving
        if self.vel.length() > 0.2:
            # The faster the car goes, the less it can turn
            turn_speed_factor = 1 - (self.vel.length() / (self.max_speed * 1.5))
            turn_speed_factor = max(0.2, turn_speed_factor) # Clamp to a minimum turn speed
            
            self.angle += steering_input * self.turn_rate * turn_speed_factor

        # --- 2. Acceleration / Braking ---
        heading = pygame.math.Vector2(math.cos(math.radians(self.angle)), 
                                      -math.sin(math.radians(self.angle)))
        
        # Check if braking or accelerating
        is_braking = (keys[pygame.K_DOWN] and self.vel.dot(heading) > 0) or \
                     (keys[pygame.K_UP] and self.vel.dot(heading) < 0)

        if is_braking:
            # Apply braking force
            brake_vec = self.vel.normalize() * self.brake_power
            # Ensure brakes can't reverse the car's velocity, only stop it
            if self.vel.length() > brake_vec.length():
                self.vel -= brake_vec
            else:
                self.vel = pygame.math.Vector2(0, 0)
        else:
            # Apply engine force (acceleration)
            self.vel += heading * self.engine_power * acceleration_input
        
        current_tile = (int(self.pos.x / TILE_SIZE), int(self.pos.y / TILE_SIZE))
        self.visited_tiles.add(current_tile)

        self.steps += 1
        self.distance += self.vel.magnitude() 

        if self.steps > MAX_STEPS_PER_CAR:
            self.alive = False
        
       
        # --- 3. Traction and Drift ---
        # This is the core of the drift physics.
        # We find the velocity component parallel to the car's heading.
        dot_product = self.vel.dot(heading)
        vel_in_heading_dir = dot_product * heading
        
        # We find the velocity component perpendicular to the car's heading (the "drift" velocity).
        vel_perpendicular = self.vel - vel_in_heading_dir
        
        # We apply traction to the perpendicular velocity, "pulling" it back in line.
        # A low self.traction value means we *keep* most of the perpendicular velocity,
        # resulting in a drift.
        self.vel = vel_in_heading_dir + (vel_perpendicular * (1.0 - self.traction))


        # --- 4. Apply Friction and Cap Speed ---
        # Apply natural friction
        self.vel *= self.friction
        
        # Cap speed
        current_speed = self.vel.length()
        if current_speed > self.max_speed:
            self.vel.scale_to_length(self.max_speed)
        elif acceleration_input < 0 and not is_braking: # Reversing
            if current_speed > abs(self.max_reverse_speed):
                self.vel.scale_to_length(abs(self.max_reverse_speed))

        # --- 5. Update Position ---
        self.pos += self.vel
        if self.check_off_track(track): # Pass the track surface
            self.alive = False

        if self.alive: # Only check if not already dead
            time_since_last_check = self.steps - self.last_check_step
            
            if time_since_last_check > STALL_CHECK_INTERVAL:
                # Calculate the distance moved since the last check
                displacement = self.pos.distance_to(self.last_check_pos)
                
                if displacement < MIN_DISPLACEMENT_PER_CHECK:
                    # Car hasn't moved enough, kill it
                    # print("Car stalled, killing.") # Optional: for debugging
                    self.alive = False
                
                # Reset for the next check period
                self.last_check_step = self.steps
                self.last_check_pos = self.pos.copy()
        
        # --- 6. Update Visuals ---
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=self.pos)

    # -----------------------------------------------------------------
    # --- FIXED FUNCTION: check_off_track ---
    # -----------------------------------------------------------------
    def check_off_track(self, track_surface):
        """Check if the car's center is on the white (off-track) area."""
        try:
            # Get the color of the pixel at the car's center
            pixel_color = track_surface.get_at((int(self.pos.x), int(self.pos.y)))
            
            # Check if the color is white (or very close to it)
            if pixel_color[0] > 200 and pixel_color[1] > 200 and pixel_color[2] > 200:
                # Car is on the grass
                return True # Return True (is off-track)
            return False # On-track

        except IndexError:
            # Car is off the screen, which counts as a crash.
            return True # Return True (is off-track)
    
    # -----------------------------------------------------------------
    # --- FIXED FUNCTION: cast_rays ---
    # -----------------------------------------------------------------
    def cast_rays(self):
        readings = []
        self.ray_endpoints = [] # <-- FIX 1: Clear the list every frame
        start = self.pos
        base_angle = self.angle
        spread = 120

        for i in range(RAY_COUNT):
            
            # <-- FIX 2: Check for RAY_COUNT == 1 BEFORE division
            if RAY_COUNT == 1:
                ray_angle = base_angle
            else:
                ray_angle = base_angle - spread / 2 + (spread / (RAY_COUNT - 1)) * i 

            rad = math.radians(ray_angle)            
            dx = math.cos(rad)
            dy = -math.sin(rad)
            
            dist = 0.0
            hit = False
            for d in range(1, RAY_LENGTH):
                x = int(start.x + dx * d)
                y = int(start.y + dy * d)

                # Check for hitting the screen edge
                if x < 0 or x >= track.get_width() or y < 0 or y >= track.get_height():
                    dist = d 
                    hit = True
                    break
                
                # Check for hitting the grass (non-road)
                if not is_on_road(x, y): # This function is now fixed
                    dist = d
                    hit = True
                    break

            if not hit:
                dist = RAY_LENGTH
            
            end_x = start.x + dx * dist
            end_y = start.y + dy * dist
            self.ray_endpoints.append((end_x, end_y))

            readings.append(dist / RAY_LENGTH)

        return readings


    def draw(self, surface):
        if not self.alive:
            return
            
        surface.blit(self.image, self.rect)

        for end_point in self.ray_endpoints:
            # Draw a line from the car's position to the ray's end-point
            pygame.draw.line(surface, GREEN, self.pos, end_point, 1)

# --- Main Game Loop ---
# -----------------------------------------------------------------
# --- FIXED FUNCTION: eval_genomes ---
# -----------------------------------------------------------------
def eval_genomes(genomes, config):
    nets = []
    cars = []

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        cars.append(Car(START_POS[0], START_POS[1]))
        genome.fitness = 0 

    run = True

    while run and any(car.alive for car in cars):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        screen.blit(track, (0, 0))

        for i, car in enumerate(cars):
                if not car.alive:
                    continue

                inputs = car.cast_rays()
                
                # <-- FIX 1: Use .length() for velocity vector
                inputs.append(car.vel.length() / car.max_speed)
                
                outputs = nets[i].activate(inputs)
                
                # <-- FIX 2: Use > 0 check instead of round()
                up = 1 if outputs[0] > 0 else 0
                down = 1 if outputs[1] > 0 else 0
                left = 1 if outputs[2] > 0 else 0
                right = 1 if outputs[3] > 0 else 0

                # <-- FIX 3: Use defaultdict(int)
                keys = defaultdict(int) 
                
                keys[pygame.K_UP] = up
                keys[pygame.K_DOWN] = down
                keys[pygame.K_RIGHT] = right
                keys[pygame.K_LEFT] = left
                
                car.update(keys)
    
                car.draw(screen)
                
                exploration_bonus = len(car.visited_tiles) * 100
                speed_bonus = car.distance * 2 

                genomes[i][1].fitness = speed_bonus + exploration_bonus

        pygame.display.flip()
        clock.tick(60)


def run(config_file):
    TOTAL_GENERATIONS = 1000

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    
    # --- Check for existing checkpoints ---
    latest_checkpoint = None
    checkpoint_files = glob.glob('neat-checkpoint-*') # Find all checkpoint files
    if checkpoint_files:
        # Sort files by generation number (e.g., neat-checkpoint-9)
        checkpoint_files.sort(key=lambda f: int(f.split('-')[-1]))
        latest_checkpoint = checkpoint_files[-1]
        
    if latest_checkpoint:
        print(f"*** Resuming training from checkpoint: {latest_checkpoint} ***")
        p = Checkpointer.restore_checkpoint(latest_checkpoint)
    else:
        print("*** Starting new training session ***")
        p = neat.Population(config)
    
    # Add reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    # This will save a checkpoint every 5 generations
    checkpointer = Checkpointer(generation_interval=5, 
                                time_interval_seconds=None, 
                                filename_prefix='neat-checkpoint-')
    p.add_reporter(checkpointer)
    
    # --- Define node names ONCE outside the loop ---
    # Update this if you change RAY_COUNT
    node_names = {
        -1: 'Ray 1', -2: 'Ray 2', -3: 'Ray 3', 
        -4: 'Ray 4', -5: 'Ray 5', -6: 'Ray 6',
        -7: 'Ray 7', -8: 'Ray 8', -9: 'Ray 9',
        -10: 'Speed',
        0: 'UP',  1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'
    }
    
    # --- Create a directory for generational graphs ---
    graph_dir = "generational-graphs"
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    print(f"Saving generation graphs to: {graph_dir}/")
    
    
    generations_left = TOTAL_GENERATIONS - p.generation
    
    # --- Run one generation at a time ---
    if generations_left > 0:
        print(f"--- Running for {generations_left} more generations (Current: {p.generation}, Target: {TOTAL_GENERATIONS}) ---")
        
        # We manually loop for the remaining generations
        for i in range(generations_left):
            # Run for ONE generation
            winner = p.run(eval_genomes, 1) # p.run returns the best genome of the generation 

        # After the loop, get the best genome from all stats
        winner = stats.best_genome()
        print('\nBest genome (from stats):\n{!s}'.format(winner))

    else:
        print(f"--- Already trained for {p.generation} generations. ---")
        winner = stats.best_genome() # Get best from stats
        print('\nBest genome (from loaded checkpoint):\n{!s}'.format(winner))


    # --- Save the FINAL winner ---
    print("\nSaving final best network...")
    visualize.draw_net(config, winner, True, 
                            node_names=node_names, 
                            filename="winner-net.gv")
    
    

if __name__ == "__main__": 
    local_dir = os.path.dirname(__file__) 
    config_path = os.path.join(local_dir, "config-feedforward.txt")  
    
    parser = argparse.ArgumentParser(description = "Run NEAT car training or simulation.")
    parser.add_argument(
        "--train",
        action = "store_true",  
        help = "If set, run the training process. Otherwise, run the best winner."
    )
    args = parser.parse_args()
    

    if args.train: 
        if running: # Only run if assets loaded
            print("--- Starting in TRAINING mode ---")
            run(config_path)
        else:
            print("Exiting due to asset loading error.")
    else: 
        print("--- Starting in TESTING mode ---")
        print("Note: Testing mode is not yet implemented.")
        # To implement testing:
        # 1. Load the 'winner' genome (e.g., from a .pkl file)
        # 2. Create a network from that genome and config
        # 3. Create one car
        # 4. Run a new game loop that feeds that car's inputs to the network
        #    and updates the car, similar to eval_genomes but for only one car.
        # run_winner(config_path, "winner-genome.pkl")
