from collections import defaultdict
import pickle
import pygame
import math
import neat
import glob
from neat import Checkpointer
import os
import visualize
import argparse

pygame.init()
WIDTH, HEIGHT = 1200, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Python Drift Car")
RAY_COUNT = 9
RAY_LENGTH = 250
MAX_STEPS_PER_CAR = 40000
TILE_SIZE = 20
STALL_CHECK_INTERVAL = 120
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
GREY = (100, 100, 100)
GREEN = (0, 200, 0)
clock = pygame.time.Clock()
FPS = 60
running = True
START_POS = (781, 507)
START_ZONE = pygame.Rect(770, 480, 30, 60)
CHECKPOINT_ZONE = pygame.Rect(550, 177, 30, 60)

try:
    track = pygame.image.load('data/track.png').convert()
    if track.get_size() != (WIDTH, HEIGHT):
        print(f"Error: 'track.png' must be {WIDTH}x{HEIGHT} pixels.")
        running = False
except FileNotFoundError:
    print("Error: 'track.png' not found.")
    running = False

def is_on_road(x, y):
    w, h = track.get_size()
    if x < 0 or x >= w or y < 0 or y >= h:
        return False
    try:
        color = track.get_at((int(x), int(y)))
        return color[0] < 100
    except IndexError:
        return False

class Car:
    def __init__(self, start_x, start_y):
        self.original_image = pygame.Surface((30, 15), pygame.SRCALPHA)
        self.original_image.fill(RED)
        pygame.draw.rect(self.original_image, GREY, (20, 0, 10, 15))
        self.pos = pygame.math.Vector2(start_x, start_y)
        self.vel = pygame.math.Vector2(0, 0)
        self.angle = 0
        self.distance = 0.0
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(start_x, start_y))
        self.steps = 0
        self.alive = True
        self.visited_tiles = set()
        start_tile = (int(self.pos.x / TILE_SIZE), int(self.pos.y / TILE_SIZE))
        self.visited_tiles.add(start_tile)
        self.max_speed = 8
        self.max_reverse_speed = -3
        self.engine_power = 0.15
        self.brake_power = 0.3
        self.turn_rate = 3
        self.friction = 0.98
        self.traction = 0.08
        self.last_check_step = 0
        self.last_check_pos = self.pos.copy()
        self.last_visited_count = len(self.visited_tiles)
        self.ray_endpoints = []
        self.checkpoint_reached = False
        self.lap_start_step = None
        self.best_lap_steps = None
        self.completed_laps = 0

    def update(self, keys, test = False):
        if not self.alive:
            return
        acceleration_input = 0
        if keys[pygame.K_UP]:
            acceleration_input = 1
        if keys[pygame.K_DOWN]:
            acceleration_input = -1
        steering_input = 0
        if keys[pygame.K_LEFT]:
            steering_input = 1
        if keys[pygame.K_RIGHT]:
            steering_input = -1
        if self.vel.length() > 0.2:
            turn_speed_factor = 1 - (self.vel.length() / (self.max_speed * 1.5))
            turn_speed_factor = max(0.2, turn_speed_factor)
            self.angle += steering_input * self.turn_rate * turn_speed_factor
        heading = pygame.math.Vector2(math.cos(math.radians(self.angle)), -math.sin(math.radians(self.angle)))
        is_braking = (keys[pygame.K_DOWN] and self.vel.dot(heading) > 0) or (keys[pygame.K_UP] and self.vel.dot(heading) < 0)
        if is_braking:
            brake_vec = self.vel.normalize() * self.brake_power
            if self.vel.length() > brake_vec.length():
                self.vel -= brake_vec
            else:
                self.vel = pygame.math.Vector2(0, 0)
        else:
            self.vel += heading * self.engine_power * acceleration_input
        current_tile = (int(self.pos.x / TILE_SIZE), int(self.pos.y / TILE_SIZE))
        self.visited_tiles.add(current_tile)
        self.steps += 1
        self.distance += self.vel.magnitude()
        if self.steps > MAX_STEPS_PER_CAR:
            self.alive = False
        dot_product = self.vel.dot(heading)
        vel_in_heading_dir = dot_product * heading
        vel_perpendicular = self.vel - vel_in_heading_dir
        self.vel = vel_in_heading_dir + (vel_perpendicular * (1.0 - self.traction))
        self.vel *= self.friction
        current_speed = self.vel.length()
        if current_speed > self.max_speed:
            self.vel.scale_to_length(self.max_speed)
        elif acceleration_input < 0 and not is_braking:
            if current_speed > abs(self.max_reverse_speed):
                self.vel.scale_to_length(abs(self.max_reverse_speed))
        self.pos += self.vel
        if self.check_off_track(track):
            self.alive = False
        if self.alive:
            time_since_last_check = self.steps - self.last_check_step
            if time_since_last_check > STALL_CHECK_INTERVAL:
                new_tiles_count = len(self.visited_tiles) - self.last_visited_count
                if new_tiles_count <= 0 and not test:
                    self.alive = False
                self.last_check_step = self.steps
                self.last_check_pos = self.pos.copy()
                self.last_visited_count = len(self.visited_tiles)
        self.check_lap_progress()
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=self.pos)

    def check_off_track(self, track_surface):
        try:
            pixel_color = track_surface.get_at((int(self.pos.x), int(self.pos.y)))
            if pixel_color[0] > 200 and pixel_color[1] > 200 and pixel_color[2] > 200:
                return True
            return False
        except IndexError:
            return True

    def cast_rays(self):
        readings = []
        self.ray_endpoints = []
        start = self.pos
        base_angle = self.angle
        spread = 120
        for i in range(RAY_COUNT):
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
                if x < 0 or x >= track.get_width() or y < 0 or y >= track.get_height():
                    dist = d
                    hit = True
                    break
                if not is_on_road(x, y):
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

    def check_lap_progress(self):
        car_rect = pygame.Rect(self.pos.x - 5, self.pos.y - 5, 10, 10)
        if not self.checkpoint_reached and CHECKPOINT_ZONE.colliderect(car_rect):
            self.checkpoint_reached = True
            self.lap_start_step = self.steps
        elif self.checkpoint_reached and START_ZONE.colliderect(car_rect):
            lap_steps = self.steps - self.lap_start_step if self.lap_start_step else None
            if lap_steps and (self.best_lap_steps is None or lap_steps < self.best_lap_steps):
                self.best_lap_steps = lap_steps
                self.completed_laps += 1
            self.checkpoint_reached = False
            self.lap_start_step = None

    def draw(self, surface):
        if not self.alive:
            return
        surface.blit(self.image, self.rect)
        for end_point in self.ray_endpoints:
            pygame.draw.line(surface, GREEN, self.pos, end_point, 1)

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
        pygame.draw.rect(screen, (0, 255, 255), START_ZONE, 2)
        pygame.draw.rect(screen, (255, 255, 0), CHECKPOINT_ZONE, 2)
        for i, car in enumerate(cars):
            if not car.alive:
                continue
            inputs = car.cast_rays()
            inputs.append(car.vel.length() / car.max_speed)
            outputs = nets[i].activate(inputs)
            up = down = left = right = 0
            if outputs[0] > 0 or outputs[1] > 0:
                if outputs[0] > outputs[1]:
                    up = 1
                else:
                    down = 1
            if outputs[2] > 0 or outputs[3] > 0:
                if outputs[2] > outputs[3]:
                    left = 1
                else:
                    right = 1
            keys = defaultdict(int)
            keys[pygame.K_UP] = up
            keys[pygame.K_DOWN] = down
            keys[pygame.K_RIGHT] = right
            keys[pygame.K_LEFT] = left
            car.update(keys)
            car.draw(screen)
            lap_reward = 0
            if car.best_lap_steps:
                lap_reward = 1000 / car.best_lap_steps
            progress_reward = len(car.visited_tiles) 
            genomes[i][1].fitness = lap_reward
        pygame.display.flip()
        clock.tick(60)

def run(config_file):
    TOTAL_GENERATIONS = 300
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    latest_checkpoint = None
    checkpoint_files = glob.glob('neat-checkpoint-*')
    if checkpoint_files:
        checkpoint_files.sort(key=lambda f: int(f.split('-')[-1]))
        latest_checkpoint = checkpoint_files[-1]
    if latest_checkpoint:
        print(f"*** Resuming training from checkpoint: {latest_checkpoint} ***")
        p = Checkpointer.restore_checkpoint(latest_checkpoint)
    else:
        print("*** Starting new training session ***")
        p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    checkpointer = Checkpointer(generation_interval=5, filename_prefix='neat-checkpoint-')
    p.add_reporter(checkpointer)
    node_names = {
        -1: 'Ray 1', -2: 'Ray 2', -3: 'Ray 3',
        -4: 'Ray 4', -5: 'Ray 5', -6: 'Ray 6',
        -7: 'Ray 7', -8: 'Ray 8', -9: 'Ray 9',
        -10: 'Speed',
        0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'
    }
    graph_dir = "generational-graphs"
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    generations_left = TOTAL_GENERATIONS - p.generation
    if generations_left > 0:
        print(f"--- Running for {generations_left} more generations (Current: {p.generation}, Target: {TOTAL_GENERATIONS}) ---")
        for i in range(generations_left):
            winner = p.run(eval_genomes, 1)
        winner = stats.best_genome()
        print('\nBest genome (from stats):\n{!s}'.format(winner))
    else:
        print(f"--- Already trained for {p.generation} generations. ---")
        winner = stats.best_genome()
        print('\nBest genome (from loaded checkpoint):\n{!s}'.format(winner))
    print("\nSaving final best network...")
    visualize.draw_net(config, winner, True, node_names=node_names, filename="winner-net.gv")
    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("Winner saved to winner.pkl")

def run_winner(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    if not os.path.exists("winner.pkl"):
        print("Error: 'winner.pkl' not found. Train first to create it.")
        return
    with open("winner.pkl", "rb") as f:
        winner = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    car = Car(START_POS[0], START_POS[1])
    print("Loaded winner. Running autonomous test...")
    global running
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return
        screen.blit(track, (0, 0))
        pygame.draw.rect(screen, (0, 255, 255), START_ZONE, 2)
        pygame.draw.rect(screen, (255, 255, 0), CHECKPOINT_ZONE, 2)
        if car.alive:
            inputs = car.cast_rays()
            inputs.append(car.vel.length() / car.max_speed)
            outputs = net.activate(inputs)
            up = down = left = right = 0
            if outputs[0] > 0 or outputs[1] > 0:
                if outputs[0] > outputs[1]:
                    up = 1
                else:
                    down = 1
            if outputs[2] > 0 or outputs[3] > 0:
                if outputs[2] > outputs[3]:
                    left = 1
                else:
                    right = 1
            keys = defaultdict(int)
            keys[pygame.K_UP] = up
            keys[pygame.K_DOWN] = down
            keys[pygame.K_LEFT] = left
            keys[pygame.K_RIGHT] = right
            car.update(keys, test = True)
        car.draw(screen)
        pygame.display.flip()
        clock.tick(FPS)
    pygame.quit()

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    parser = argparse.ArgumentParser(description="Run NEAT car training or simulation.")
    parser.add_argument("--train", action="store_true", help="If set, run the training process. Otherwise, run the best winner.")
    args = parser.parse_args()
    if args.train:
        if running:
            print("--- Starting in TRAINING mode ---")
            run(config_path)
        else:
            print("Exiting due to asset loading error.")
    else:
        print("--- Starting in TESTING mode ---")
        run_winner(config_path)

