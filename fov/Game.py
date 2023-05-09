import math
import pygame as pg
import random
import neat, time


pg.init()
font = pg.font.SysFont(None, 24)

fps = 120

class Game:
    def __init__(self, screen, width, height) -> None:
        self.screen = screen
        self.width = width
        self.height = height
        self.clock = pg.time.Clock()
        self.running = True
        # place player at the bottom of the screen
        self.player = pg.Rect(0, 0, 20, 20)
        self.player.center = (width / 2, height/2)
        self.speed = 15# max speed
        self.range = 200
        self.old_positions = [self.player.center]

        while True:
            self.target = pg.Rect(random.randint(60, width-60), random.randint(60, height-60), 70, 70)
            if self.player.colliderect(self.target):
                continue
            break

        self.frames = 0

        self.net = None
        self.genome = None

        self.record = False
        self.follow_mouse = False
        self.record_frames = []

    def run(self):
        while self.running:
            self.loop()
            if self.frames == 325:
                print(self.genome.fitness)
                break

    def train(self, genome, config):
        start = time.time()
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.run()
        return time.time() - start

    def draw(self):

        # if self.follow_mouse and pg.mouse.get_focused():
        #     self.target.center = pg.mouse.get_pos()
        # elif self.frames % 200 == 0:
        #     self.target.center = (random.randint(30, self.width-30), random.randint(30, self.height-30))

        pg.draw.circle(self.screen, (255, 255, 255), self.player.center, self.range, 1)
        pg.draw.rect(self.screen, (255, 255, 255), self.target)
        pg.draw.rect(self.screen, (50, 120, 255), self.player)

        global fps
        # fps may become too much at times to render
        try:
            frame_rate_render = font.render(
                f"FPS: {round(self.clock.get_fps())}/{fps}", True, (255, 255, 255)
            )
            self.screen.blit(frame_rate_render, (10, 30))
        except:
            pass

    def loop(self):
        global fps
        self.screen.fill((0, 0, 0))
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            # use arrow keys to change fps

            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.running = False
                if event.key == pg.K_RIGHT:
                    fps = 100000
                if event.key == pg.K_LEFT:
                    fps = 120

        self.get_input()
        self.draw()

        # updae screen every few frames or if fps is low
        if fps == 120:
            pg.display.flip()
            self.clock.tick(fps)

        if self.record:
            # get the surface as a numpy array and swap axes. Much, much faster than writing an image
            self.record_frames.append(pg.surfarray.array3d(self.screen).swapaxes(0, 1))
        self.frames += 1

    def get_input(self):
        # get dx and dy from target to player pythagorean theorem
        dist = math.sqrt((self.player.x - self.target.x)**2 + (self.player.y - self.target.y)**2)
        sensed = False
        inps = [dist, self.player.x, self.player.y]
        # if the target is within range, give coordinates and indicate that it is sensed
        if dist < self.range:
            inps.append(1)
            inps.append(self.target.x - self.player.x)
            inps.append(self.target.y - self.player.y)
            sensed = True
        else:
            inps.append(0)
            inps.append(0)
            inps.append(0)
        # feed inputs into the neural network
        output = self.net.activate(inps)
        dec = output.index(max(output))
        # move up
        if dec == 0:
            self.player.y -= self.speed
        # move down
        elif dec == 1:
            self.player.y += self.speed
        # move left
        elif dec == 2:
            self.player.x -= self.speed
        # move right
        elif dec == 3:
            self.player.x += self.speed

        # if the player collides with the target, reward it 
        if self.player.colliderect(self.target):
            self.genome.fitness += 4
            return

        if not (self.player.x > 0 and self.player.x < self.width and self.player.y > 0 and self.player.y < self.height):
            self.genome.fitness -= 4
            return

        # discourage the player from staying in the same spot
        hit = False
        for pos in self.old_positions:
            pg.draw.rect(self.screen, (255, 255, 255), (pos[0], pos[1], 5, 5))
            if self.player.collidepoint(pos):
                self.genome.fitness -= 0.13
                hit = True
        if not hit:
            self.genome.fitness += 0.4
        self.old_positions.append(self.player.center)
        # only keep the last 15 positions
        if len(self.old_positions) > 15:
            self.old_positions.pop(0)

        # reward based on new distance to target 
        new_dist = math.sqrt((self.player.x - self.target.x)**2 + (self.player.y - self.target.y)**2)
        new_dist = new_dist if new_dist >= 0 else 1

        if new_dist < dist:
            if sensed:
                self.genome.fitness += 0.9
            else:
                self.genome.fitness += 0.2
        else:
            if sensed:
                self.genome.fitness -= 1.2
            else:
                self.genome.fitness -= 0.2
        
        
    def test(self, genome, config, follow_mouse=False) -> list:
        global fps
        fps = 120
        self.follow_mouse = follow_mouse
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.record = True
        self.run()
        print(self.genome.fitness)
        return self.record_frames
